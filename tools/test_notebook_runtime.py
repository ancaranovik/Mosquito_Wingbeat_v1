"""Regression checks for independent notebook kernels; no network or training."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from project_paths import REPO_ROOT, ProjectPaths
import notebook_runtime


class NotebookRuntimeTests(unittest.TestCase):
    def bootstrap(self, name):
        notebook = json.loads((REPO_ROOT / 'notebooks/current' / name).read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code'
                      and 'from notebook_runtime import initialize' in ''.join(c['source']))
        return source.split('from project_paths import REPO_ROOT, ProjectPaths, resolve_path, logical_relative')[0]

    def test_all_three_find_04d_checkout_from_new_kernel(self):
        target = Path('/content/edge-ai')
        for name in ('04A_train_logmel.ipynb', '04B_train_edge_mfe.ipynb', '04C_compare_models.ipynb'):
            with self.subTest(notebook=name), patch.dict(os.environ), \
                 patch.object(Path, 'cwd', return_value=Path(tempfile.gettempdir())), \
                 patch.object(Path, 'is_file', side_effect=lambda p: p in (target / 'tools/project_paths.py', target / 'tools/notebook_runtime.py'), autospec=True), \
                 patch.object(sys, 'path', list(sys.path)), \
                 patch('notebook_runtime.initialize', return_value=ProjectPaths()) as initialize, \
                 patch('subprocess.run') as command:
                os.environ.pop('EDGEAI_REPO_ROOT', None)
                scope = {}
                exec(self.bootstrap(name), scope)
                initialize.assert_called_once_with(target)
                command.assert_not_called()

    def test_fresh_colab_kernel_clones_without_running_training(self):
        google = types.ModuleType('google')
        colab = types.ModuleType('google.colab')
        google.colab = colab
        target = Path('/content/edge-ai').resolve()
        state = {'cloned': False}
        def clone(*args, **kwargs): state['cloned'] = True
        def is_file(p): return state['cloned'] and p == target / 'tools/notebook_runtime.py'
        with patch.dict(os.environ), patch.dict(sys.modules, {'google': google, 'google.colab': colab}), \
             patch.object(Path, 'cwd', return_value=Path(tempfile.gettempdir())), \
             patch.object(Path, 'is_file', autospec=True, side_effect=is_file), \
             patch.object(Path, 'exists', return_value=False), \
             patch.object(sys, 'path', list(sys.path)), \
             patch('notebook_runtime.initialize', return_value=ProjectPaths()) as initialize, \
             patch('subprocess.run', side_effect=clone) as command:
            os.environ.pop('EDGEAI_REPO_ROOT', None)
            exec(self.bootstrap('04B_train_edge_mfe.ipynb'), {})
            self.assertEqual(command.call_args.args[0], ['git', 'clone', 'https://github.com/ancaranovik/Mosquito_Wingbeat_v1.git', str(target)])
            initialize.assert_called_once_with(target)

    def test_existing_incomplete_directory_is_preserved(self):
        google = types.ModuleType('google'); colab = types.ModuleType('google.colab'); google.colab = colab
        with patch.dict(os.environ), patch.dict(sys.modules, {'google': google, 'google.colab': colab}), \
             patch.object(Path, 'is_file', return_value=False), patch.object(Path, 'exists', return_value=True), \
             patch('subprocess.run') as command:
            os.environ.pop('EDGEAI_REPO_ROOT', None)
            with self.assertRaisesRegex(RuntimeError, 'exists but is not an EdgeAI checkout'):
                exec(self.bootstrap('04C_compare_models.ipynb'), {})
            command.assert_not_called()

    def test_initialize_mounts_drive_in_its_own_kernel(self):
        google = types.ModuleType('google'); colab = types.ModuleType('google.colab'); google.colab = colab
        colab.drive = Mock()
        with patch.dict(os.environ), patch.dict(sys.modules, {'google': google, 'google.colab': colab}), \
             patch('notebook_runtime.is_colab_runtime', return_value=True), \
             patch.object(Path, 'is_dir', return_value=False), patch('os.chdir') as chdir, \
             patch('notebook_runtime.configure', return_value=ProjectPaths()) as configure:
            notebook_runtime.initialize(REPO_ROOT)
            colab.drive.mount.assert_called_once_with('/content/drive')
            configure.assert_called_once_with(install=False)
            chdir.assert_called_once_with(REPO_ROOT)

    def test_local_initialize_keeps_external_storage_and_does_not_install(self):
        storage = REPO_ROOT / '_local_only/test_external_storage'
        with patch.dict(os.environ, {'EDGEAI_DATA_ROOT': str(storage)}), \
             patch('notebook_runtime.is_colab_runtime', return_value=False), patch('os.chdir'), \
             patch('notebook_runtime.configure', return_value=ProjectPaths(data=storage)) as configure:
            paths = notebook_runtime.initialize(REPO_ROOT)
            self.assertEqual(paths.data, storage)
            configure.assert_called_once_with(install=False)

    def test_switching_checkout_requires_kernel_restart(self):
        with patch('notebook_runtime.configure') as configure, \
             self.assertRaisesRegex(RuntimeError, 'Restart this notebook kernel'):
            notebook_runtime.initialize(REPO_ROOT / '_local_only/another_checkout')
        configure.assert_not_called()


if __name__ == '__main__':
    unittest.main()
