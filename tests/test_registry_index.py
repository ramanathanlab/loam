import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


candidate = load('registry_index_candidate', ROOT / 'skills/hpc-deployment/scripts/hpc_agent.py')
legacy = load('registry_index_legacy', ROOT / 'skills/hpc-deployment/scripts/hpc_agent.py')


class RegistryIndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='loam-registry-index-')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.registry = self.base / 'batches/registry'
        self.registry.mkdir(parents=True)
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('HPC_BATCH_ROOT', None)
        os.environ.pop('HPC_TOOL_ROOT', None)
        reset = patch.object(candidate, '_REGISTRY_INDEX', candidate._RegistryIndex())
        reset.start()
        self.addCleanup(reset.stop)

    def entry(self, number, hypothesis='h1', **changes):
        timestamp = datetime(2026, 11, 18, tzinfo=timezone.utc) + timedelta(microseconds=number)
        value = {'batch_id': f'b-{number:012x}', 'created_at': timestamp.isoformat(),
                 'labels': {'hypothesis_id': hypothesis, 'run_id': 'r001-01', 'plan_id': 'p1'},
                 'total': 1, 'source_inputs': 'inputs.jsonl', 'tool': 'science', 'tool_version': 1,
                 'expects': ['expected.json']}
        value.update(changes)
        return value

    def write(self, value):
        candidate._write_json(self.registry / (value['batch_id'] + '.json'), value)

    def assert_equivalent(self):
        for hypothesis in [None, '', 'h1', 'h2', 'missing']:
            for plan in [None, '', 'p1', 'missing']:
                self.assertEqual(candidate._list_batches(str(self.base), plan, hypothesis),
                                 legacy._list_batches(str(self.base), plan, hypothesis))
        self.assertEqual(candidate._tool_usage(str(self.base)), legacy._tool_usage(str(self.base)))

    def test_warm_existing_entries_match_disk_queries(self):
        for n in range(1, 20):
            self.write(self.entry(n, 'h1' if n % 2 else 'h2',
                                  last_status={'unfinished': 0, 'failed': n % 3}))
        self.assertTrue(candidate.enable_registry_index(str(self.base)))
        self.assert_equivalent()

    def test_new_and_updated_entries_are_visible_without_rescans(self):
        self.assertTrue(candidate.enable_registry_index(str(self.base)))
        value = self.entry(1)
        self.write(value)
        value['last_status'] = {'unfinished': 0, 'failed': 0}
        self.write(value)
        with patch.object(candidate, '_registry_entries', side_effect=AssertionError('global rescan')):
            self.assertEqual(candidate._list_batches(str(self.base), hypothesis_id='h1')[0]['last_status'],
                             value['last_status'])
            self.assertEqual(candidate._list_batches(str(self.base), hypothesis_id='missing'), [])
            self.assertEqual(candidate._tool_usage(str(self.base))[0], {('science', 1): 1})
        self.assert_equivalent()

    def test_no_registry_file_reads_after_warm(self):
        self.write(self.entry(1))
        candidate.enable_registry_index(str(self.base))
        original = Path.read_text
        registry = self.registry
        def guarded(path, *args, **kwargs):
            if path.parent == registry:
                raise AssertionError('registry file opened by indexed query')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', guarded):
            self.assertEqual(len(candidate._list_batches(str(self.base), hypothesis_id='h1')), 1)
            self.assertEqual(candidate._tool_usage(str(self.base))[0], {('science', 1): 1})

    def test_label_and_tool_changes_remove_old_index_membership(self):
        value = self.entry(1)
        self.write(value)
        candidate.enable_registry_index(str(self.base))
        value['labels']['hypothesis_id'] = 'h2'
        value['tool'] = 'different_science'
        value['tool_version'] = 2
        self.write(value)
        self.assertEqual(candidate._list_batches(str(self.base), hypothesis_id='h1'), [])
        self.assert_equivalent()

    def test_returned_nested_data_does_not_mutate_index(self):
        value = self.entry(1, last_status={'unfinished': 0, 'failed': 0})
        self.write(value)
        candidate.enable_registry_index(str(self.base))
        result = candidate._list_batches(str(self.base))
        result[0]['labels']['hypothesis_id'] = 'someone_else'
        result[0]['last_status']['failed'] = 99
        self.assert_equivalent()

    def test_json_key_conversion_matches_disk(self):
        value = self.entry(1)
        value['labels']['extra'] = {1: 'one'}
        candidate.enable_registry_index(str(self.base))
        self.write(value)
        self.assertEqual(candidate._list_batches(str(self.base))[0]['labels']['extra'], {'1': 'one'})
        self.assert_equivalent()

    def test_row_output_failure_is_not_hidden_by_index(self):
        value = self.entry(1)
        self.write(value)
        folder = self.base / 'batches' / value['batch_id'] / 'run_one'
        folder.mkdir(parents=True)
        candidate._write_json(folder / 'status.json', {'returncode': 0, 'missing_outputs': ['expected.json']})
        (folder / 'done').touch()
        candidate.enable_registry_index(str(self.base))
        status = candidate._batch_status(value['batch_id'], str(self.base))
        self.assertFalse(status['ok'])
        self.assertEqual(status['failed'], 1)
        self.assertEqual(candidate._tool_usage(str(self.base))[1], {})
        (folder / 'expected.json').write_text('{}')
        candidate._write_json(folder / 'status.json', {'returncode': 0})
        self.assertTrue(candidate._batch_status(value['batch_id'], str(self.base))['ok'])
        self.assertEqual(candidate._tool_usage(str(self.base))[1][('science', 1)][1], value['batch_id'])
        self.assert_equivalent()

    def test_failure_before_replace_keeps_disk_and_index_old(self):
        value = self.entry(1)
        self.write(value)
        candidate.enable_registry_index(str(self.base))
        changed = self.entry(1, total=99)
        with patch.object(candidate.os, 'replace', side_effect=OSError('test atomic failure')):
            with self.assertRaises(OSError):
                self.write(changed)
        self.assertEqual(json.loads((self.registry / (value['batch_id'] + '.json')).read_text())['total'], 1)
        self.assert_equivalent()

    def test_concurrent_replacements_match_final_disk_state(self):
        candidate.enable_registry_index(str(self.base))
        def publish(n):
            value = self.entry(1, 'h1' if n % 2 else 'h2', total=n + 1,
                               last_status={'unfinished': 0, 'failed': n % 3})
            self.write(value)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(publish, range(120)))
        self.assert_equivalent()

    def test_disabled_mode_observes_external_disk_updates(self):
        self.write(self.entry(1))
        value = self.entry(2)
        (self.registry / (value['batch_id'] + '.json')).write_text(json.dumps(value))
        self.assert_equivalent()

    def test_foreign_pid_ignores_inherited_snapshot(self):
        self.write(self.entry(1))
        candidate.enable_registry_index(str(self.base))
        value = self.entry(2)
        (self.registry / (value['batch_id'] + '.json')).write_text(json.dumps(value))
        pid = os.getpid()
        with patch.object(candidate.os, 'getpid', return_value=pid + 123):
            self.assertIsNone(candidate._REGISTRY_INDEX.entries(str(self.base)))
            self.assertEqual(len(candidate._list_batches(str(self.base))), 2)

    def test_torn_json_skipped_as_in_disk_iterator(self):
        self.write(self.entry(1))
        (self.registry / 'torn.json').write_text('{')
        self.assertTrue(candidate.enable_registry_index(str(self.base)))
        self.assert_equivalent()

    def test_unsupported_metadata_disables_index_without_hiding_entry(self):
        self.write(self.entry(1))
        (self.registry / 'unsupported.json').write_text('{"other": 1}')
        self.assertFalse(candidate.enable_registry_index(str(self.base)))
        self.assertIsNone(candidate._REGISTRY_INDEX.entries(str(self.base)))
        self.assert_equivalent()

    def test_multiple_campaigns_and_aliases_are_isolated(self):
        self.write(self.entry(1))
        other = self.base / 'other'
        other_registry = other / 'batches/registry'
        other_registry.mkdir(parents=True)
        candidate._write_json(other_registry / 'b-000000000002.json', self.entry(2))
        candidate.enable_registry_index(str(self.base))
        candidate.enable_registry_index(str(other))
        self.assertEqual(candidate._list_batches(str(other))[0]['batch_id'], 'b-000000000002')
        alias = self.base / 'alias'
        alias.symlink_to(other, target_is_directory=True)
        self.assertEqual(candidate._list_batches(str(alias)), candidate._list_batches(str(other)))
        self.assert_equivalent()

    def test_second_precision_ties_preserve_membership_and_newest_time(self):
        for n in range(1, 8):
            self.write(self.entry(n, created_at='2026-11-18T00:00:00+00:00',
                                  last_status={'unfinished': 0, 'failed': 0}))
        candidate.enable_registry_index(str(self.base))
        old = legacy._list_batches(str(self.base))
        new = candidate._list_batches(str(self.base))
        self.assertEqual({v['batch_id'] for v in old}, {v['batch_id'] for v in new})
        self.assertEqual(candidate._tool_usage(str(self.base))[0], legacy._tool_usage(str(self.base))[0])
        # Existing API promises newest first, not a tie-breaker between same-second births.
        self.assertEqual(candidate._tool_usage(str(self.base))[1][('science', 1)][0],
                         legacy._tool_usage(str(self.base))[1][('science', 1)][0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
