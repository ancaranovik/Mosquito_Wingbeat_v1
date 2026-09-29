"""Storage isolation and experiment safety checks; no training."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from project_paths import ProjectPaths, REPO_ROOT, resolve_path, require_mutable_output, logical_relative
from experiment_runner import validate_config, run


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.storage = REPO_ROOT / '_local_only/test_external_drive'
        self.env = patch.dict(os.environ, {'EDGEAI_DATA_ROOT': str(self.storage)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_manifest_split_audio_and_cache_mapping(self):
        paths = ProjectPaths.from_env()
        self.assertEqual(paths.RAW_ROOT, self.storage / 'fixed/raw/HumBugDB')
        split = 'data/manifests/current/core_single_4species_frozen_split.csv'
        self.assertEqual(resolve_path(REPO_ROOT, split), paths.SPLIT_ROOT / Path(split).name)
        self.assertEqual(logical_relative(resolve_path(REPO_ROOT, split)), Path(split))
        self.assertEqual(paths.resolve('data/stage04_cache/03A/features.npy'), paths.CACHE_ROOT / '03A/features.npy')
        self.assertFalse(paths.CACHE_ROOT.exists())  # no silent legacy fallback

    def test_fixed_writes_rejected(self):
        paths = ProjectPaths.from_env()
        for root in (paths.RAW_ROOT, paths.CACHE_ROOT, paths.BASELINE_ROOT, paths.MANIFEST_ROOT,
                     paths.SPLIT_ROOT, paths.data / 'fixed/new_data'):
            with self.subTest(root=root), self.assertRaises(PermissionError):
                require_mutable_output(root / 'attempt.json')
        self.assertEqual(require_mutable_output(paths.experiment('exp_test') / 'results/a.json'),
                         paths.experiment('exp_test') / 'results/a.json')

    def test_path_traversal_rejected(self):
        paths = ProjectPaths.from_env()
        for name in ('../fixed', 'exp_../fixed', 'a', '/tmp/exp_test', 'exp_x/y'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                paths.experiment(name)
        with self.assertRaises(ValueError):
            paths.resolve('../outside')

    def test_scientific_overrides_rejected(self):
        import json
        config = json.loads((REPO_ROOT / 'configs/experiments/baseline_reuse.json').read_text())
        self.assertTrue(validate_config(config).name.startswith('exp_'))
        config['scientific_overrides'] = {'learning_rate': 0.003}
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_existing_experiment_never_overwritten(self):
        with patch.object(Path, 'exists', return_value=True), patch('experiment_runner.smoke') as smoke:
            with self.assertRaises(FileExistsError):
                run(REPO_ROOT / 'configs/experiments/baseline_reuse.json')
            smoke.assert_not_called()

    def test_external_root_does_not_make_local_science_writable(self):
        for relative in ('data/metadata/release.csv', 'data/stage04_cache/03A/features.npy'):
            with self.assertRaises(PermissionError):
                require_mutable_output(REPO_ROOT / relative)

    def test_direct_training_entry_requires_reserved_experiment(self):
        import experiment_training
        with patch.object(experiment_training, 'EXPERIMENT_ID', None):
            with self.assertRaisesRegex(RuntimeError, 'reserve a unique experiment ID'):
                experiment_training.run_branch('03A')


if __name__ == '__main__':
    unittest.main()
