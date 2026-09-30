"""Configure a cloned checkout after mounting Drive; no training or data writes."""
import os
from pathlib import Path
import subprocess
import sys
from project_paths import REPO_ROOT, COLAB_DATA_ROOT, ProjectPaths, is_colab_runtime


def configure(install=False):
    if sys.version_info < (3, 11):
        raise RuntimeError('Python 3.11+ required; accepted Colab run used 3.13.15')
    os.environ['EDGEAI_REPO_ROOT'] = str(REPO_ROOT)
    if is_colab_runtime():
        if not Path('/content/drive/MyDrive').is_dir():
            raise FileNotFoundError('Mount Google Drive before configuring the runtime')
        os.environ.setdefault('EDGEAI_DATA_ROOT', str(COLAB_DATA_ROOT))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    paths = ProjectPaths.from_env()
    if not paths.CACHE_ROOT.is_dir():
        raise FileNotFoundError(f'Mount Drive and upload both accepted caches first: {paths.CACHE_ROOT}')
    if install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(REPO_ROOT / 'requirements-colab.txt')], check=True)
    print('Repository:', REPO_ROOT)
    print('Data:', paths.data)
    print('Immutable baseline:', paths.BASELINE_ROOT)
    print('Future results:', paths.EXPERIMENT_ROOT / '<experiment_id>')
    return paths
