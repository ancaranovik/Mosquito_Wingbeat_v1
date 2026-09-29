from pathlib import Path
import hashlib,json,time
root=Path.cwd()
assert not (root/"docs/reorganization/before.json").exists(), "Baseline already exists; do not overwrite"
records=[]
for p in root.rglob('*'):
    if not p.is_file() or '.git' in p.parts or p.is_relative_to(root/'docs/reorganization'): continue
    with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    records.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':digest})
(root/'docs/reorganization/before.json').write_text(json.dumps(records,indent=2)+'\n')
print('Hashed',len(records),'files')
