"""Storage isolation and experiment safety checks; no training."""
import os
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from project_paths import ProjectPaths, REPO_ROOT, resolve_path, require_mutable_output, logical_relative
from experiment_runner import validate_config, run
# Load extension modules before patch.dict(sys.modules): restoring that mock must
# not unload NumPy/PyTorch extensions first imported inside a fake-Colab context.
import cache_consumer
import experiment_training_v2


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


    def test_local_bootstrap_preserves_configured_storage(self):
        from colab_bootstrap import configure
        with patch('colab_bootstrap.is_colab_runtime', return_value=False), \
             patch('project_paths.is_colab_runtime', return_value=False), \
             patch.object(Path, 'is_dir', return_value=True):
            self.assertEqual(configure(install=False).data, self.storage)

    def test_colab_rejects_local_data_root(self):
        with patch('project_paths.is_colab_runtime', return_value=True):
            with self.assertRaisesRegex(ValueError, 'Colab data must'):
                ProjectPaths.from_env()

    def test_colab_defaults_to_mounted_drive(self):
        from project_paths import COLAB_DATA_ROOT
        with patch.dict(os.environ), patch('project_paths.is_colab_runtime', return_value=True):
            os.environ.pop('EDGEAI_DATA_ROOT', None)
            paths = ProjectPaths.from_env()
            self.assertEqual(paths.data, COLAB_DATA_ROOT)
            self.assertEqual(paths.EXPERIMENT_ROOT, COLAB_DATA_ROOT / 'experiments')

    def test_legacy_cache_staging_rejects_fixed_storage(self):
        from stage04_data import cache_staging
        with self.assertRaises(PermissionError):
            with cache_staging(ProjectPaths.from_env().CACHE_ROOT, '03A'):
                self.fail('Fixed cache was writable')

    def test_00_promotion_is_blocked_even_if_enabled(self):
        import ast, json
        notebook = json.loads((REPO_ROOT / 'notebooks/current/00_prepare_dataset.ipynb').read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'WRITE_CANONICAL = False' in ''.join(c['source']))
        tree = ast.parse(source)
        guarded_write = next(n for n in tree.body if isinstance(n, ast.If))
        target = ProjectPaths.from_env().MANIFEST_ROOT / 'blocked.csv'
        with self.assertRaises(PermissionError):
            exec(compile(ast.Module(body=[guarded_write], type_ignores=[]), '<promotion guard>', 'exec'),
                 {'WRITE_CANONICAL': True, 'verified_payloads': {target: b'never written'}})


    def test_04d_colab_mount_clone_pull_and_cuda_routing(self):
        import json, sys, types
        from unittest.mock import Mock
        notebook = json.loads((REPO_ROOT / 'notebooks/current/04D_experiments.ipynb').read_text(encoding='utf-8'))
        # Dependency installation must precede any live-kernel Torch import.
        setup_index = next(i for i, c in enumerate(notebook['cells']) if 'paths = configure(' in ''.join(c['source']))
        self.assertFalse(any('import torch' in ''.join(c['source']) for c in notebook['cells'][:setup_index]
                             if c['cell_type'] == 'code'))
        google = types.ModuleType('google')
        colab = types.ModuleType('google.colab')
        colab.drive = Mock()
        google.colab = colab
        url = 'https://github.com/ancaranovik/Mosquito_Wingbeat_v1.git'
        for exists in (False, True):
            with self.subTest(existing_checkout=exists), \
                 patch.dict(sys.modules, {'google': google, 'google.colab': colab}), \
                 patch.dict(os.environ, {'EDGEAI_REPO_ROOT': '/content/edge-ai'}), \
                 patch('project_paths.is_colab_runtime', return_value=False), \
                 patch.object(sys, 'path', list(sys.path)), \
                 patch('os.chdir'), patch.object(Path, 'exists', return_value=exists), \
                 patch('subprocess.run') as run_command, \
                 patch('subprocess.check_output', side_effect=[url, ''] if exists else []) as read_command, \
                 contextlib.redirect_stdout(io.StringIO()):
                scope = {}
                exec(next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code' and 'REPO_URL =' in ''.join(c['source'])), scope)
                colab.drive.mount.assert_called_with('/content/drive')
                self.assertTrue(scope['IN_COLAB'])
                args = run_command.call_args.args[0]
                self.assertEqual(args[:3], ['git', 'pull', '--ff-only'] if exists else ['git', 'clone', url])
                scope['paths'] = ProjectPaths(repo=REPO_ROOT, data=Path('/content/drive/MyDrive/EdgeAI'))
                exec(next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code' and 'device = ' in ''.join(c['source'])), scope)
                self.assertEqual(run_command.call_args.args[0][-3:], ['smoke', '--device', 'cuda'])
                before = run_command.call_count
                config_fixture = cache_consumer.read_json(REPO_ROOT / 'configs/experiments/control_v2.json')
                with patch('cache_consumer.read_json', return_value=config_fixture):
                    exec(next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code' and 'RUN_TRAINING = False' in ''.join(c['source'])), scope)
                self.assertFalse(scope['RUN_TRAINING'])
                self.assertEqual(run_command.call_count, before)

    def test_cuda_smoke_fails_before_cache_access_without_gpu(self):
        from experiment_runner import smoke
        with patch('torch.cuda.is_available', return_value=False), \
             patch('cache_consumer.prepare_caches') as caches:
            with self.assertRaisesRegex(RuntimeError, 'Select a Colab GPU'):
                smoke('cuda')
            caches.assert_not_called()


if __name__ == '__main__':
    unittest.main()
