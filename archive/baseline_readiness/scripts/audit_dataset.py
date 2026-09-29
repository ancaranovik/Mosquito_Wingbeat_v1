"""Pre-training provenance/audio audit. Does not window, augment, or train audio.

Frozen candidate partitions are source-family-disjoint, NOT verified individual-disjoint.
Run once after every listed public file has downloaded. Existing freeze blocks reruns.
"""
import sys,pathlib,json,re,hashlib,csv,collections,io,math
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python_libs'))
import numpy as np
import soundfile as sf
from scipy.signal import correlate
SPECIES={'Ae.Aegypti':'Aedes aegypti','Ae.Albopictus':'Aedes albopictus','An.Dirus':'Anopheles dirus','Cx.Quin':'Culex quinquefasciatus'}
LABELS={k:v.replace(' ','_').lower() for k,v in SPECIES.items()}
def sha(b):return hashlib.sha256(b).hexdigest()
def writecsv(name,rows,fields=None):
    with (ROOT/'manifests'/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)
def main():
    freeze=ROOT/'manifests'/'freeze.json'
    if freeze.exists():raise SystemExit('Existing frozen manifests: do not silently repartition. Create a reviewed new version instead.')
    inventory=json.loads((ROOT/'sources'/'drive_inventory.json').read_text())
    missing=[r['path'] for r in inventory if not pathlib.Path(r['local_path']).exists()]
    if missing:raise SystemExit(f'{len(missing)} listed audio files missing: full audit not yet ready')
    rows=[]; arrays={}; pcm={}; groups=collections.defaultdict(list)
    for item in inventory:
        p=pathlib.Path(item['local_path']); data=p.read_bytes(); x,sr=sf.read(p,dtype='int16',always_2d=True); info=sf.info(p)
        if len(x)!=info.frames:raise ValueError('Truncated file '+p.name)
        name=p.name
        row=dict(source_identifier=item['id'],filename=name,kind=item['kind'],species='',label='',sex='',sex_token='',day_token='',temperature_token='',apparatus_token='',microphone_variant='',cut_token='',individual_id='',session_id='',original_recording_id='',independent_group_identifier='',group_evidence='',duration_seconds=round(len(x)/sr,6),sample_rate=sr,channels=x.shape[1],subtype=info.subtype,frames=len(x),file_sha256=sha(data),pcm_sha256=sha(str((sr,x.shape)).encode()+x.tobytes()),partition='',candidate_partition='',exclusion_reason='',baseline_duration_eligible=len(x)>=sr,local_path=str(p.relative_to(ROOT)).replace('\\','/'))
        if item['kind']=='mosquito':
            m=re.fullmatch(r'([^_]+)_([0-9]+[FM])\s*_([^_]+)_([^_]+)_([^_]+)_(?:(LowMic)_)?(cut\d+)_16bit\.wav',name,re.I)
            if not m:raise ValueError('Unparsed filename: '+name)
            sp,sex,day,temp,app,mic,cut=m.groups()
            family=f'{sp}_{sex.strip()}_{day}_{temp}_{app}'.lower()
            row.update(species=SPECIES[sp],label=LABELS[sp],sex=sex.strip()[-1],sex_token=sex.strip(),day_token=day,temperature_token=temp,apparatus_token=app,microphone_variant='LowMic' if mic else 'unmarked',cut_token=cut.lower(),independent_group_identifier=family,group_evidence='inferred_filename_source_family; merges cuts and microphone variants; individual/session identity UNKNOWN')
            arrays[item['id']]=x[:,0]; pcm[item['id']]=x.tobytes()
            groups[family].append(row)
        else:
            row.update(label='environmental_noise_NOT_CORE_CLASS',independent_group_identifier='noise_quarantine_all',group_evidence='recording/session relations unresolved; all noise held outside species training',partition='excluded',candidate_partition='excluded',exclusion_reason='Noise is not one of the four candidate closed-set species; no augmentation or detector in this milestone')
        rows.append(row)
    # Exact file/PCM duplicates, plus bounded exact overlap screening.
    duplicate_rows=[]
    for key in ('file_sha256','pcm_sha256'):
        clusters=collections.defaultdict(list)
        for r in rows:clusters[r[key]].append(r)
        for digest,rr in clusters.items():
            if len(rr)>1:
                for r in rr:duplicate_rows.append(dict(check=key,digest=digest,source_identifier=r['source_identifier'],filename=r['filename']))
    writecsv('duplicates.csv',duplicate_rows,['check','digest','source_identifier','filename'])
    # Three 100 ms anchors per file, searched at every byte offset in other files.
    # This finds some exact reuse, not all shifted/gain-scaled/near-duplicate audio.
    mosquitoes=[r for r in rows if r['kind']=='mosquito']; overlaps=[]
    for i,a in enumerate(mosquitoes):
        aa=arrays[a['source_identifier']]; n=round(a['sample_rate']*.1)
        anchors=[]
        for start in sorted(set([0,max(0,(len(aa)-n)//2),max(0,len(aa)-n)])):
            block=aa[start:start+n]
            if len(block)==n and len(np.unique(block))>=32:anchors.append((start,block.tobytes()))
        for b in mosquitoes[i+1:]:
            if a['sample_rate']!=b['sample_rate']:continue
            target=pcm[b['source_identifier']]
            for start,block in anchors:
                pos=target.find(block)
                if pos>=0 and pos%2==0:
                    overlaps.append(dict(source_a=a['source_identifier'],source_b=b['source_identifier'],file_a=a['filename'],file_b=b['filename'],a_offset_samples=start,b_offset_samples=pos//2,anchor_samples=n,cross_family=a['independent_group_identifier']!=b['independent_group_identifier']))
                    break
    writecsv('exact_overlap_screen.csv',overlaps,['source_a','source_b','file_a','file_b','a_offset_samples','b_offset_samples','anchor_samples','cross_family'])
    if any(x['cross_family'] for x in overlaps):
        raise SystemExit('Cross-family exact reuse found. Review/merge groups before freezing. Audit evidence saved.')
    # Explicitly proposed compromise: four recoverable families/class -> 2/1/1.
    assignment={}
    for species in sorted(SPECIES.values()):
        sg=[g for g,rr in groups.items() if rr[0]['species']==species]
        sexes={s:sorted([g for g in sg if groups[g][0]['sex']==s],key=lambda g:sha(('20260923|'+g).encode())) for s in ['F','M']}
        if sorted(map(len,sexes.values()))!=[2,2]:raise ValueError('Unexpected family support; review split rather than invent allocations')
        assignment[sexes['F'][0]]='training'; assignment[sexes['M'][0]]='training'
        held=sorted([sexes['F'][1],sexes['M'][1]],key=lambda g:sha(('holdout|'+g).encode()))
        assignment[held[0]]='validation'; assignment[held[1]]='testing'
    for r in mosquitoes:
        r['partition']=assignment[r['independent_group_identifier']]
        r['candidate_partition']=r['partition']
        reasons=['HOLD: source-family grouping is not verified individual/session independence']
        if not r['baseline_duration_eligible']:reasons.append('shorter than original 1-second context; no padding/looping/concatenation authorized')
        r['exclusion_reason']='; '.join(reasons)
    writecsv('split_manifest.csv',rows)
    group_rows=[]
    for g,rr in sorted(groups.items()):
        group_rows.append(dict(group_identifier=g,species=rr[0]['species'],sex=rr[0]['sex'],day_token=rr[0]['day_token'],files=len(rr),duration_seconds=round(sum(r['duration_seconds'] for r in rr),6),files_at_least_1s=sum(r['baseline_duration_eligible'] for r in rr),microphone_variants=';'.join(sorted(set(r['microphone_variant'] for r in rr))),verified_individual_count='unknown',verified_session_count='unknown',partition=assignment[g]))
    writecsv('group_manifest.csv',group_rows)
    summary={}
    for species in sorted(SPECIES.values()):
        rr=[r for r in mosquitoes if r['species']==species]
        durations=np.array([r['duration_seconds'] for r in rr])
        summary[species]=dict(files=len(rr),duration_seconds=round(float(durations.sum()),6),duration_min=float(durations.min()),duration_median=float(np.median(durations)),duration_max=float(durations.max()),files_at_least_1s=sum(r['baseline_duration_eligible'] for r in rr),sex_files=dict(collections.Counter(r['sex'] for r in rr)),recoverable_source_families=len({r['independent_group_identifier'] for r in rr}),verified_individuals=None,verified_sessions=None,partitions={p:dict(groups=sum(g['species']==species and g['partition']==p for g in group_rows),files=sum(r['partition']==p for r in rr),files_at_least_1s=sum(r['partition']==p and r['baseline_duration_eligible'] for r in rr)) for p in ['training','validation','testing']})
    result=dict(species=summary,total_files=len(rows),mosquito_files=len(mosquitoes),noise_files=len(rows)-len(mosquitoes),formats=dict(collections.Counter(f"{r['sample_rate']} Hz / {r['channels']} ch / {r['subtype']}" for r in rows)),exact_duplicate_rows=len(duplicate_rows),exact_overlap_pairs=len(overlaps),grouping_status='inferred_source_families_only',training_decision='NO-GO for final experiment',selected_species=[],candidate_species=sorted(SPECIES.values()),split_ratio_by_family='50/25/25; closest nonempty three-way integer allocation for four families/class; 60/20/20 not exactly achievable')
    (ROOT/'dataset_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    for p in ['training','validation','testing']:
        rr=[r for r in mosquitoes if r['partition']==p]
        writecsv(p+'_files_HOLD.csv',rr)
    checks=dict(inventory_complete=len(rows)==1336 and len(mosquitoes)==1324,group_disjoint=all(len({r['partition'] for r in rr})==1 for rr in groups.values()),identities_not_fabricated=all(r['individual_id']==r['session_id']==r['original_recording_id']=='' for r in mosquitoes),no_cross_family_exact_overlap=not any(x['cross_family'] for x in overlaps))
    assert all(checks.values()),checks
    hashes={p.name:sha(p.read_bytes()) for p in (ROOT/'manifests').glob('*.csv')}
    freeze.write_text(json.dumps(dict(version='source-family-candidate-v1',date='2026-09-23',seed='20260923',status='HOLD_NO_GO_NOT_VERIFIED_INDEPENDENT',checks=checks,manifest_sha256=hashes),indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
