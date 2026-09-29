"""Read-only scientific/path checks; never execute the research pipeline."""
from pathlib import Path
from collections import Counter,defaultdict
import ast,csv,hashlib,json,os
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def verify():
    baseline=json.loads((ROOT/'docs/reorganization/before.json').read_text())
    moves=json.loads((ROOT/'docs/reorganization/moves.json').read_text())
    checked=0; deleted=[]; changed=[]
    for r in baseline:
        old=r['path']; new=old
        if old.startswith('tools/maintenance/'): continue # tooling created during cleanup
        for m in moves:
            if old==m['from'] or old.startswith(m['from']+'/'):
                new=m['to']+old[len(m['from']):]; break
        if old.startswith('build/'):
            assert not (ROOT/new).exists(); deleted.append(old); continue
        p=ROOT/new
        assert p.is_file(), f'Missing preserved file: {old} -> {new}'
        if sha(p)!=r['sha256']:
            backup=ROOT/'archive/reorganization_originals'/Path(old).name
            assert backup.is_file() and sha(backup)==r['sha256'], f'Unexpected change: {old}'
            changed.append(new)
        checked+=1
    current=ROOT/'data/manifests/current'
    core=rows(current/'core_single_4species_metadata.csv')
    split=rows(current/'core_single_4species_frozen_split.csv')
    grid=rows(current/'core_example_grid_v2_stride15360_ref15600_candidates.csv')
    assert len(core)==len(split)==1898
    assert len({r['id'] for r in core})==1898
    assert len({r['name'] for r in split})==1189
    assert Counter(r['split'] for r in split)=={'train':1335,'validation':269,'test':294}
    sources=defaultdict(set)
    for r in split: sources[r['name']].add(r['split'])
    assert all(len(v)==1 for v in sources.values())
    lookup={r['id']:r for r in split}
    assert len(grid)==32645 and len({r['example_uid'] for r in grid})==32645
    for r in grid:
        assert all(r[k]==lookup[r['id']][k] for k in ('name','species','split'))
        assert int(r['example_stride_samples'])==15360
        assert int(r['reference_context_samples'])==15600
        assert int(r['start_sample_16k'])==int(r['example_index'])*15360
        assert int(r['reference_end_sample_16k'])-int(r['start_sample_16k'])==15600
    assert all((ROOT/'data/audio'/f"{r['id']}.wav").is_file() for r in split)
    audit=json.loads((ROOT/'data/audits/current/core_example_grid_v2_stride15360_ref15600_audit.json').read_text())
    assert sha(current/'core_example_grid_v2_stride15360_ref15600_candidates.csv')==audit['manifest_sha256']
    for r in audit['clips']:
        assert sha(ROOT/'data/audio'/f"{r['id']}.wav")==r['wav_sha256']
    # Compile every code cell, and evaluate only path declarations/root discovery.
    path_checks=[]
    for nb in sorted((ROOT/'notebooks/current').glob('*.ipynb')):
        n=json.loads(nb.read_text(encoding='utf-8'))
        old=json.loads((ROOT/'archive/reorganization_originals'/nb.name).read_text(encoding='utf-8'))
        for a,b in zip(n['cells'],old['cells']):
            assert a.get('outputs')==b.get('outputs') and a.get('execution_count')==b.get('execution_count')
        for cwd in (ROOT,nb.parent):
            env={'Path':Path,'__builtins__':__builtins__}
            os.chdir(cwd)
            for cell in n['cells']:
                if cell['cell_type']!='code': continue
                src=''.join(cell['source']); tree=ast.parse(src)
                for node in tree.body:
                    use=isinstance(node,ast.FunctionDef) and node.name in ('find_project_root','first_existing')
                    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                        name=node.targets[0].id
                        use=name in {'PROJECT','DATA_DIR','AUDIO_DIR','RAW_METADATA_PATH','METADATA_PATH','CORE_PATH','SPLIT_PATH','AUDIO_AUDIT_PATH','REPRO_DIR','OLD_MANIFEST','OUTPUT_MANIFEST','OUTPUT_AUDIT','NOTEBOOK_PATH','PDF_DIR','MANIFEST_PATH','INPUT_PATHS','EXPECTED_INPUT_SHA256','audit_dir_candidates','AUDIT_DIR'}
                    if use: exec(compile(ast.Module(body=[node],type_ignores=[]),str(nb),'exec'),env)
            assert env['PROJECT']==ROOT
            for k,v in env.items():
                if isinstance(v,Path): assert v.exists(),f'{nb.name} {k}: {v}'
            for name,expected in env.get('EXPECTED_INPUT_SHA256',{}).items():
                assert sha(ROOT/env['INPUT_PATHS'][name])==expected
            path_checks.append(f'{nb.name} from {cwd.relative_to(ROOT).as_posix()}: PASS')
    os.chdir(ROOT)
    result={'preserved_files_verified':checked,'deleted_build_files':len(deleted),'changed_with_original_backup':changed,'core_clips':1898,'sources':1189,'splits':dict(Counter(r['split'] for r in split)),'source_leakage':0,'examples':32645,'stride':15360,'reference_context':15600,'core_audio_hashes_match_grid_audit':len(audit['clips']),'path_checks':path_checks,'pipeline_executed':False}
    print(json.dumps(result,indent=2))
    return result
if __name__=='__main__': verify()
