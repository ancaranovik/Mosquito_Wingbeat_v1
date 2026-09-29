from pathlib import Path
import json,shutil
root=Path.cwd()
backup=root/'archive/reorganization_originals'
assert not backup.exists(), "Original backups already exist; do not rerun"
backup.mkdir(parents=True,exist_ok=True)
shutil.copy2(root/'README.md',backup/'README.md')
paths={
'neurips_2021_zenodo_0_0_1.csv':'data/metadata/neurips_2021_zenodo_0_0_1.csv',
'core_single_4species_metadata.csv':'data/manifests/current/core_single_4species_metadata.csv',
'core_single_4species_frozen_split.csv':'data/manifests/current/core_single_4species_frozen_split.csv',
'core_audio_audit.csv':'data/audits/current/core_audio_audit.csv',
'core_096s_candidate_windows.csv':'archive/legacy_data/core_096s_candidate_windows.csv',
'core_example_grid_v2_stride15360_ref15600_candidates.csv':'data/manifests/current/core_example_grid_v2_stride15360_ref15600_candidates.csv',
'core_example_grid_v2_stride15360_ref15600_audit.json':'data/audits/current/core_example_grid_v2_stride15360_ref15600_audit.json',
}
for p in (root/'notebooks/current').glob('*.ipynb'):
    shutil.copy2(p,backup/p.name)
    n=json.loads(p.read_text(encoding='utf-8'))
    for c in n['cells']:
        s=''.join(c['source'])
        s=s.replace('data/neurips_2021_zenodo_0_0_1.csv',paths['neurips_2021_zenodo_0_0_1.csv'])
        s=s.replace('PROJECT = Path(r"D:\\VGU-27\\Edge AI Project")','PROJECT = next((p for p in [Path.cwd(), *Path.cwd().parents]\n    if (p / "data/metadata/neurips_2021_zenodo_0_0_1.csv").is_file()), None)\nassert PROJECT is not None, "Run from the project root or its notebook directory"')
        for name,new in paths.items():
            s=s.replace('DATA_DIR / "'+name+'"','PROJECT / "'+new+'"')
        s=s.replace('DATA_DIR / "_reproduced_00"','PROJECT / "reports/reproduced_00"')
        if 'path = DATA_DIR / name' in s:
            s=s.replace('path = DATA_DIR / name','path = PROJECT / INPUT_PATHS[name]')
            s='INPUT_PATHS = '+repr({k:v for k,v in paths.items() if k in ['neurips_2021_zenodo_0_0_1.csv','core_single_4species_metadata.csv','core_single_4species_frozen_split.csv','core_audio_audit.csv','core_096s_candidate_windows.csv']})+'\n\n'+s
        s=s.replace('PROJECT / "archive" / "old_experiments" / "baseline_readiness"','PROJECT / "archive" / "baseline_readiness"')
        c['source']=s.splitlines(keepends=True)
    p.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('Updated notebook paths only; originals preserved')
