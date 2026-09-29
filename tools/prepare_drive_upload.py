"""Non-destructive Drive staging. Large files get exact, hashed upload mappings.

Existing staging files must match; nothing is overwritten or removed.
"""
import csv
import hashlib
from pathlib import Path
import shutil
from project_paths import REPO_ROOT


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mappings():
    for source, destination, purpose in (
        ('data/audio', 'fixed/raw/HumBugDB', 'Preserved HumBugDB audio; includes all local audio, not only Core'),
        ('data/metadata', 'fixed/data/metadata', 'Release metadata'),
        ('data/audits/current', 'fixed/data/audits', 'Frozen scientific acceptance evidence'),
        ('data/stage04_cache/03A', 'fixed/data/stage04_cache/03A', 'Validated immutable Log-Mel cache'),
        ('data/stage04_cache/03B', 'fixed/data/stage04_cache/03B', 'Validated immutable Edge MFE cache'),
        ('colab_result/stage04_colab_results', 'fixed/baseline/stage04', 'Eight completed accepted CUDA baseline runs'),
        ('archive/baseline_readiness/raw', 'fixed/history/archive/baseline_readiness/raw', 'Historical provenance audio; never current inputs'),
        ('archive/baseline_readiness/sources/historical_archives', 'fixed/history/archive/baseline_readiness/sources/historical_archives', 'Historical source audio archives; never current inputs'),
    ):
        base = REPO_ROOT / source
        for path in sorted(base.rglob('*')):
            if path.is_file():
                yield path, Path(destination) / path.relative_to(base), purpose
    for path in sorted((REPO_ROOT / 'data/manifests/current').glob('*')):
        if path.is_file():
            folder = 'splits' if path.name == 'core_single_4species_frozen_split.csv' else 'manifests'
            yield path, Path('fixed/data') / folder / path.name, 'Frozen population/split/segmentation identity'


def main():
    staging = REPO_ROOT / '_drive_upload/EdgeAI'
    for relative in ('fixed/raw/HumBugDB', 'fixed/data/manifests', 'fixed/data/splits',
                     'fixed/data/audits', 'fixed/data/metadata', 'fixed/data/stage04_cache/03A',
                     'fixed/data/stage04_cache/03B', 'fixed/baseline/stage04',
                     'fixed/history/archive/baseline_readiness/raw',
                     'fixed/history/archive/baseline_readiness/sources/historical_archives',
                     'experiments', 'derived', 'reports', 'temp'):
        (staging / relative).mkdir(parents=True, exist_ok=True)
    records = []
    for source, destination, purpose in mappings():
        size = source.stat().st_size
        sha = digest(source)
        # Avoid duplicating audio and large arrays; all are explicitly listed.
        copied = source.suffix.lower() not in ('.wav', '.npy', '.zip') and size <= 32 * 1024 * 1024
        target = staging / destination
        if copied:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != sha:
                    raise RuntimeError(f'Staging conflict; preserve and investigate: {target}')
            else:
                shutil.copyfile(source, target)
            if digest(target) != sha:
                raise RuntimeError(f'Copy verification failed: {target}')
        records.append({'source': str(source), 'drive_destination': 'MyDrive/EdgeAI/' + destination.as_posix(),
                        'bytes': size, 'sha256': sha, 'purpose': purpose,
                        'staged': copied, 'staging_path': str(target) if copied else ''})
    for path in (REPO_ROOT / 'docs/portability/drive_migration.csv', REPO_ROOT / '_drive_upload/drive_migration.csv'):
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    print(f'{len(records)} mapped files, {sum(r["bytes"] for r in records):,} bytes; '
          f'{sum(r["staged"] for r in records)} staged; all others explicitly mapped.')


if __name__ == '__main__':
    main()
