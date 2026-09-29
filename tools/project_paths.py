"""Storage boundaries for the frozen pipeline and new experiments.

EDGEAI_DATA_ROOT selects the EdgeAI directory, never its fixed/ child.
Without it, local reads use the original project layout. No files are moved.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DATA_ROOT = Path('/content/drive/MyDrive/EdgeAI')


@dataclass(frozen=True)
class ProjectPaths:
    repo: Path = REPO_ROOT
    data: Path | None = None

    @classmethod
    def from_env(cls, repo=REPO_ROOT):
        value = os.environ.get('EDGEAI_DATA_ROOT')
        data = Path(value).expanduser().resolve() if value else None
        if data is None and Path('/content').is_dir():
            data = COLAB_DATA_ROOT
        return cls(Path(repo).resolve(), data)

    @property
    def storage(self):
        return self.data if self.data is not None else self.repo / '_runtime'

    @property
    def RAW_ROOT(self):
        return self.data / 'fixed/raw/HumBugDB' if self.data else self.repo / 'data/audio'

    @property
    def MANIFEST_ROOT(self):
        return self.data / 'fixed/data/manifests' if self.data else self.repo / 'data/manifests/current'

    @property
    def SPLIT_ROOT(self):
        return self.data / 'fixed/data/splits' if self.data else self.repo / 'data/manifests/current'

    @property
    def CACHE_ROOT(self):
        return self.data / 'fixed/data/stage04_cache' if self.data else self.repo / 'data/stage04_cache'

    @property
    def BASELINE_ROOT(self):
        return self.data / 'fixed/baseline/stage04' if self.data else self.repo / 'colab_result/stage04_colab_results'

    @property
    def EXPERIMENT_ROOT(self):
        return self.storage / 'experiments'

    @property
    def DERIVED_ROOT(self):
        return self.storage / 'derived'

    @property
    def REPORT_ROOT(self):
        return self.storage / 'reports'

    def resolve(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Expected a repository-relative path without traversal')
        text = relative.as_posix()
        if not self.data:
            return self.repo / relative
        if text == 'data/manifests/current/core_single_4species_frozen_split.csv':
            return self.SPLIT_ROOT / relative.name
        mappings = {
            'data/audio': self.RAW_ROOT,
            'data/manifests/current': self.MANIFEST_ROOT,
            'data/stage04_cache': self.CACHE_ROOT,
            'data/audits/current': self.data / 'fixed/data/audits',
            'data/metadata': self.data / 'fixed/data/metadata',
        }
        for prefix, target in mappings.items():
            if text == prefix or text.startswith(prefix + '/'):
                return target / text[len(prefix):].lstrip('/')
        return self.repo / relative

    def experiment(self, experiment_id):
        if not re.fullmatch(r'exp_[A-Za-z0-9][A-Za-z0-9_-]{0,95}', experiment_id):
            raise ValueError('Experiment ID must start exp_ and contain only letters, digits, _ or -')
        target = (self.EXPERIMENT_ROOT / experiment_id).resolve()
        if not target.is_relative_to(self.EXPERIMENT_ROOT.resolve()):
            raise ValueError('Experiment path escapes storage')
        return target


def resolve_path(repo, relative):
    return ProjectPaths.from_env(repo).resolve(relative)


def logical_relative(path, repo=REPO_ROOT):
    """Keep historical provenance identifiers independent of physical storage."""
    paths = ProjectPaths.from_env(repo)
    path = Path(path).resolve()
    for prefix in ('data/audio', 'data/manifests/current', 'data/stage04_cache',
                   'data/audits/current', 'data/metadata'):
        base = paths.resolve(prefix)
        if path.is_relative_to(base):
            return Path(prefix) / path.relative_to(base)
    if path.is_relative_to(paths.SPLIT_ROOT):
        return Path('data/manifests/current') / path.relative_to(paths.SPLIT_ROOT)
    return path.relative_to(paths.repo)


def require_mutable_output(path):
    paths = ProjectPaths.from_env()
    target = Path(path).resolve()
    protected = [paths.RAW_ROOT, paths.MANIFEST_ROOT, paths.SPLIT_ROOT, paths.CACHE_ROOT,
                 paths.BASELINE_ROOT, paths.resolve('data/audits/current'),
                 paths.resolve('data/metadata'), paths.repo / 'data',
                 paths.repo / 'reports/stage04', paths.repo / 'colab_result']
    if paths.data:
        protected.append(paths.data / 'fixed')
    if any(target.is_relative_to(p.resolve()) for p in protected):
        raise PermissionError(f'Immutable scientific storage: {target}')
    return target
