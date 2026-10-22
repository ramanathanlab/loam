"""Layer 0 provenance classifier (C4): observed input lineage, domain-blind."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / ".loam" / "hooks" / "provenance.py"


def _load():
    spec = importlib.util.spec_from_file_location("loam_provenance", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A real GLP1R-length ECD-like sequence (pasted from memory would look like this).
HARDCODED_SEQ = "A" * 120


class ProvenanceLayer0Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.prov = _load()

    def _payload(self, code: str, tool_name: str = "mcp__hpc__run_code") -> dict:
        return {"tool_name": tool_name, "arguments": {"payload": code}}

    def test_inline_only_flags_a_hardcoded_sequence(self) -> None:
        code = f'target = "{HARDCODED_SEQ}"\nfold(target)'
        lineage = self.prov.classify(self._payload(code))
        self.assertEqual(lineage["verdict"], "inline_only")
        self.assertTrue(lineage["inline_literals"])
        self.assertEqual(lineage["inline_literals"][0]["looks_like"], "protein_sequence_like")
        self.assertFalse(lineage["fetched"])

    def test_fetched_sequence_is_not_inline_only(self) -> None:
        code = (
            "import requests\n"
            "seq = requests.get('https://rest.uniprot.org/uniprotkb/P43220.fasta').text\n"
        )
        lineage = self.prov.classify(self._payload(code))
        self.assertEqual(lineage["verdict"], "fetched")
        self.assertIn("fetched", lineage["origins"])

    def test_fetch_plus_inline_literal_is_mixed(self) -> None:
        code = (
            "import requests\n"
            "ref = requests.get('https://rest.uniprot.org/uniprotkb/P43220.fasta').text\n"
            f'binder = "{HARDCODED_SEQ}"\n'
        )
        lineage = self.prov.classify(self._payload(code))
        self.assertEqual(lineage["verdict"], "mixed")
        self.assertIn("fetched", lineage["origins"])
        self.assertIn("inline_literal", lineage["origins"])

    def test_filesystem_read(self) -> None:
        code = "seq = open('/flare/FRAME-IDP/avasan/hpc-run/target.fasta').read()\n"
        lineage = self.prov.classify(self._payload(code))
        self.assertEqual(lineage["verdict"], "filesystem_read")

    def test_small_molecule_smiles_is_flagged_generically(self) -> None:
        # Generalization beyond protein: a short SMILES literal still flags.
        code = 'lig = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine\nscore(lig)'
        lineage = self.prov.classify(self._payload(code))
        self.assertEqual(lineage["verdict"], "inline_only")
        kinds = {lit["looks_like"] for lit in lineage["inline_literals"]}
        self.assertIn("smiles_like", kinds)

    def test_prose_and_short_args_are_not_literals(self) -> None:
        code = (
            'print("Running the decisive fold now and reading the metrics")\n'
            'path = "out_dir"\n'
            'mode = "xpu:0"\n'
        )
        lineage = self.prov.classify(self._payload(code))
        self.assertEqual(lineage["inline_literals"], [])
        self.assertEqual(lineage["verdict"], "none")

    def test_non_execution_tool_is_ignored(self) -> None:
        self.assertIsNone(
            self.prov.classify({"tool_name": "delegate_task", "arguments": {"goal": "x" * 200}})
        )

    def test_run_batch_paths_count_as_filesystem_or_derived(self) -> None:
        payload = {
            "tool_name": "mcp__hpc__run_batch",
            "arguments": {"script_path": "/flare/x/script.py", "inputs_path": "/flare/x/in.jsonl",
                          "batch_id": "b-abc123def456"},
        }
        lineage = self.prov.classify(payload)
        self.assertIn("derived_ref", lineage["origins"])

    def test_record_writes_jsonl_and_is_observe_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "tool_name": "mcp__hpc__run_code",
                "loam_state_dir": tmp,
                "loam_run_id": "run-1",
                "agent_role": "code-generator",
                "arguments": {"payload": f'target = "{HARDCODED_SEQ}"'},
            }
            self.prov.record(payload)
            lines = (Path(tmp) / "provenance.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row["verdict"], "inline_only")
            self.assertEqual(row["record_version"], "loam_provenance_v0")
            self.assertEqual(row["agent_role"], "code-generator")

        # No state dir and no env var -> silently does nothing, never raises.
        self.prov.record({"tool_name": "mcp__hpc__run_code", "arguments": {"payload": "x"}})


if __name__ == "__main__":
    unittest.main()
