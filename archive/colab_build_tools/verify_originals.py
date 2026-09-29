from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time

BUILD = Path(__file__).resolve().parent
ROOT = BUILD.parent
before = json.loads((BUILD / 'original_snapshot.json').read_text(encoding='utf-8'))
files = before['files']


def verify(item):
    relative, expected = item
    try:
        with (ROOT / relative).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        return relative if actual != expected else None
    except OSError as exc:
        return f'{relative}: {exc}'


changed = []
with ThreadPoolExecutor(max_workers=12) as pool:
    futures = [pool.submit(verify, item) for item in files.items()]
    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        if result:
            changed.append(result)
        if i % 5000 == 0:
            print(f'Original files verified: {i}/{len(files)}', flush=True)
assert not changed, changed
report = {'status': 'PASS', 'original_files_sha256_verified_unchanged': len(files),
          'changed_original_files': [], 'original_snapshot_sha256': hashlib.sha256(
              (BUILD / 'original_snapshot.json').read_bytes()).hexdigest(),
          'unreadable_preserved_cache_directories': ['data/stage04_cache/preserved-before-audit-20260929/03A',
                                                    'data/stage04_cache/preserved-before-audit-20260929/03B'],
          'unreadable_directory_note': 'Legacy preserved cache folders excluded from sandbox inventory; no writes targeted them.',
          'training_executed_during_export': False, 'features_regenerated': False}
(BUILD / 'stage04_colab_bundle/provenance/original_files_verification.json').write_text(
    json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(f'PASS: {len(files)} original files are byte-identical', flush=True)
