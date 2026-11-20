"""Provenance guards for compact manifests; run with the scientific Python."""
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import gemmi
    import numpy as np
    import scipy
except ImportError as exc:
    raise unittest.SkipTest('Scientific Python with gemmi/numpy required') from exc

ROOT = Path(__file__).resolve().parents[1]
MODULE = Path(os.environ.get('COMPACT_AUDIT_MODULE_PATH', ROOT / 'scripts/audit_scale20_campaign.py'))
spec = importlib.util.spec_from_file_location('compact_audit_under_test', MODULE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CompactModelMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.row = Path(self.tmp.name) / 'row'
        (self.row / 'fold').mkdir(parents=True)
        (self.row / 'models').mkdir()
        self.original = self.row / 'fold/pred.model_idx_0.cif'
        self.copy = self.row / 'models/pred.model_idx_0.cif'
        self.score = self.row / 'fold/scores.model_idx_0.npz'
        structure = gemmi.Structure()
        model = gemmi.Model('1')
        for name, count, residue_name, y in [('A', 205, 'ALA', 0), ('B', 60, 'GLY', 5)]:
            chain = gemmi.Chain(name)
            for i in range(1, count + 1):
                residue = gemmi.Residue()
                residue.name = residue_name
                residue.seqid = gemmi.SeqId(i, ' ')
                atom = gemmi.Atom()
                atom.name = 'CA'
                atom.element = gemmi.Element('C')
                atom.pos = gemmi.Position(i * 3.8, y, 0)
                residue.add_atom(atom)
                chain.add_residue(residue)
            model.add_chain(chain)
        structure.add_model(model)
        structure.setup_entities()
        structure.make_mmcif_document().write_file(str(self.original))
        shutil.copy2(self.original, self.copy)
        np.savez(self.score, iptm=np.array([.75]), ptm=np.array([.8]), aggregate_score=np.array([.77]))
        self.item = {'path': str(self.copy), 'sha256': audit.sha(self.copy)}
        (self.row / 'summary.json').write_text(json.dumps({'models': [self.item]}))

    def verify(self, item=None):
        records = []
        result = audit.verify_screen_model(item if item is not None else self.item,
                                          self.row, 'A' * 205, 'G' * 60, records)
        return result, records

    def test_exact_copy_uses_original_pair_without_modifying_artifacts(self):
        paths = [self.copy, self.original, self.score, self.row / 'summary.json']
        before = {str(p): audit.sha(p) for p in paths}
        result, records = self.verify()
        self.assertEqual(result['path'], str(self.copy))
        self.assertEqual(result['scores']['iptm'], .75)
        self.assertEqual(self.item['scores']['iptm'], .75)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['original_score_file'], str(self.score))
        self.assertEqual(before, {str(p): audit.sha(p) for p in paths})

    def test_regular_model_metadata_retains_existing_checks(self):
        item = audit.verify_model(self.original, 'A' * 205, 'G' * 60)
        result, records = self.verify(item)
        self.assertEqual(result, item)
        self.assertEqual(records, [])

    def test_false_reported_score_is_rejected(self):
        item = audit.verify_model(self.original, 'A' * 205, 'G' * 60)
        item['scores']['iptm'] = .1
        with self.assertRaises(AssertionError):
            self.verify(item)

    def test_different_copied_bytes_are_rejected_even_with_matching_copy_hash(self):
        self.copy.write_bytes(self.copy.read_bytes() + b'\n# changed copy\n')
        self.item['sha256'] = audit.sha(self.copy)
        with self.assertRaises(AssertionError):
            self.verify()

    def test_missing_original_score_is_not_searched_for_elsewhere(self):
        self.score.rename(self.row / 'models/scores.model_idx_0.npz')
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_cross_row_original_is_rejected(self):
        other = self.row.parent / 'other'
        other.mkdir()
        original = other / self.original.name
        self.original.rename(original)
        self.original.symlink_to(original)
        with self.assertRaises(AssertionError):
            self.verify()

    def test_different_prediction_index_is_rejected(self):
        other = self.original.with_name('pred.model_idx_1.cif')
        self.original.rename(other)
        self.original.symlink_to(other.name)
        with self.assertRaises(AssertionError):
            self.verify()

    def test_different_score_index_is_rejected(self):
        other = self.score.with_name('scores.model_idx_1.npz')
        self.score.rename(other)
        self.score.symlink_to(other.name)
        with self.assertRaises(AssertionError):
            self.verify()

    def test_other_incomplete_schemas_are_rejected(self):
        self.item['scores'] = {'iptm': .75}
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_nonfinite_original_score_is_rejected(self):
        np.savez(self.score, iptm=np.array([np.nan]), ptm=np.array([.8]), aggregate_score=np.array([.77]))
        with self.assertRaises(AssertionError):
            self.verify()

    def test_wrong_binder_sequence_is_rejected(self):
        with self.assertRaises(AssertionError):
            audit.verify_screen_model(self.item, self.row, 'A' * 205, 'A' * 60, [])


if __name__ == '__main__':
    unittest.main()
