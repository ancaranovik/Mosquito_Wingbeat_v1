"""Initialize a notebook kernel independently of another notebook's environment."""
import os
from pathlib import Path
import sys

from project_paths import REPO_ROOT, is_colab_runtime
from colab_bootstrap import configure


def initialize(repo):
    repo = Path(repo).resolve()
    if repo != REPO_ROOT:
        raise RuntimeError('Another checkout is already imported. Restart this notebook kernel and rerun its first cell.')
    os.environ['EDGEAI_REPO_ROOT'] = str(repo)
    sys.dont_write_bytecode = True
    if is_colab_runtime():
        from google.colab import drive
        if not Path('/content/drive/MyDrive').is_dir():
            drive.mount('/content/drive')
    paths = configure(install=False)
    os.chdir(repo)
    return paths
