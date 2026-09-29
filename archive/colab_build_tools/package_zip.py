from pathlib import Path
import hashlib
import json
import zipfile
import ast

BUILD = Path(__file__).resolve().parent
ROOT = BUILD.parent
PACKAGE = BUILD / 'stage04_colab_bundle'
DEST = ROOT / 'stage04_colab_bundle.zip'
assert not DEST.exists(), 'Refusing to overwrite an existing bundle'
check = json.loads((PACKAGE / 'provenance/original_files_verification.json').read_text())
assert check['status'] == 'PASS' and not check['changed_original_files']
manifest = json.loads((PACKAGE / 'bundle_manifest.json').read_text())
notebook = json.loads((PACKAGE / 'stage04_colab.ipynb').read_text())
mh = hashlib.sha256((PACKAGE / 'bundle_manifest.json').read_bytes()).hexdigest()
assert mh in ''.join(''.join(c['source']) for c in notebook['cells'])
for c in notebook['cells']:
    if c['cell_type'] == 'code':
        ast.parse(''.join(c['source']))
        assert c['outputs'] == [] and c['execution_count'] is None
files = sorted(p for p in PACKAGE.rglob('*') if p.is_file())
assert not any((PACKAGE / 'results').rglob('*.*')), 'Results must be empty'
temporary = BUILD / 'bundle.zip'
with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                     allowZip64=True) as archive:
    for p in files:
        name = p.relative_to(PACKAGE).as_posix()
        print('Compress:', name, flush=True)
        archive.write(p, name)
    for name in ('results/', 'results/runs/'):
        info = zipfile.ZipInfo(name)
        info.create_system = 3
        info.external_attr = (0o40755 << 16) | 0x10
        archive.writestr(info, '')
print('Verifying every ZIP member by SHA256 and CRC', flush=True)
with zipfile.ZipFile(temporary) as archive:
    assert len(archive.namelist()) == len(set(archive.namelist()))
    for p in files:
        name = p.relative_to(PACKAGE).as_posix()
        with p.open('rb') as f:
            expected = hashlib.file_digest(f, 'sha256').hexdigest()
        with archive.open(name) as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        assert actual == expected, name
        if name in manifest['sha256']:
            assert actual == manifest['sha256'][name], name
    assert sum(n.endswith('.ipynb') for n in archive.namelist()) == 1
temporary.replace(DEST)
with DEST.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
print(json.dumps({'zip': str(DEST), 'bytes': DEST.stat().st_size,
                  'MiB': DEST.stat().st_size / 2**20, 'sha256': digest,
                  'file_count': len(files), 'empty_directories': 2}, indent=2), flush=True)
