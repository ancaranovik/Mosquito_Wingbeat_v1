"""Validate and freeze existing memberships; no training or waveform conversion."""
from audit_humbug_four import R,Q,rows,write
import csv,json,hashlib,collections,datetime
def read(p): return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (Q/'four_class_freeze.json').exists()
c=read(Q/'four_class_split_candidate_metadata.csv'); q=json.loads((Q/'audio_qc.json').read_text())
assert len(c)==len(q)==1596 and {r['source_id'] for r in c}==set(q)=={r['id'] for r in rows}
assert len({r['source_id'] for r in c})==1596
ov=json.loads((Q/'overlap_summary.json').read_text()); assert ov['cross_original']==0
assert len({a['pcm_sha256'] for a in q.values()})==1596
assert all(a['actual_duration']>=1 and a['zero_fraction']<1 and a['clip_fraction']==0 and abs(a['duration_error_seconds'])<1/44100+1e-8 for a in q.values())
assert all(a['actual_sample_rate']==44100 and a['channels']==1 and a['subtype']=='PCM_24' for a in q.values())
final=[]
for r in c:
 a=q[r['source_id']]
 assert a['name']==r['name'] and a['species']==r['species']
 final.append(dict(**{k:v for k,v in r.items() if k!='audio_audit_status'},independent_group_identifier=r['name'],local_path=a['local_path'],actual_duration=a['actual_duration'],sample_rate=a['actual_sample_rate'],channels=a['channels'],subtype=a['subtype'],file_sha256=a['file_sha256'],pcm_sha256=a['pcm_sha256'],exclusion_reason='',status='FROZEN_FOUR_CLASS_V1'))
groups=collections.defaultdict(list)
for r in final: groups[r['name']].append(r)
assert len(groups)==975
assert all(len({r['partition'] for r in rr})==len({r['species'] for r in rr})==1 for rr in groups.values())
write(Q/'four_class_source_manifest.csv',final)
write(Q/'four_class_group_manifest.csv',[dict(name=n,species=rr[0]['species'],partition=rr[0]['partition'],clips=len(rr),seconds=sum(r['actual_duration'] for r in rr)) for n,rr in sorted(groups.items())])
for p in ['training','validation','testing']:write(Q/('four_class_'+p+'_files.csv'),[r for r in final if r['partition']==p])
summary=[]
for sp in sorted({r['species'] for r in final}):
 for p in ['training','validation','testing']:
  rr=[r for r in final if r['species']==sp and r['partition']==p]
  summary.append(dict(species=sp,partition=p,clips=len(rr),groups=len({r['name'] for r in rr}),seconds=sum(r['actual_duration'] for r in rr)))
write(Q/'final_split_summary.csv',summary)
previous=read(Q/'missing_local_audio.csv')
coverage=dict(expected=1596,present=1596,missing=0,previous_missing=len(previous),previous_missing_now_present=sum((R.parent/'data/audio'/r['source_file']).exists() if 'source_file' in r else (R.parent/'data/audio'/(r['id']+'.wav')).exists() for r in previous))
(Q/'coverage_verification.json').write_text(json.dumps(coverage,indent=2))
old=read(R/'humbug_audit/core_testing_files.csv'); old_names={r['name'] for r in old}
cross=[r for r in final if r['name'] in old_names and r['partition']!='testing']
if cross:write(Q/'secondary_test_cross_experiment_overlap.csv',cross)
stats=dict(coverage=coverage,clips=1596,groups=975,exclusions=0,exact_pcm_duplicates=0,overlap=ov,seconds=sum(a['actual_duration'] for a in q.values()),max_duration_error=max(abs(a['duration_error_seconds']) for a in q.values()),secondary_test_groups_in_four_development=len({r['name'] for r in cross}),split=summary)
(Q/'final_audit_summary.json').write_text(json.dumps(stats,indent=2))
files=['four_class_source_manifest.csv','four_class_group_manifest.csv','four_class_training_files.csv','four_class_validation_files.csv','four_class_testing_files.csv','final_split_summary.csv','audio_inventory.csv','overlap_summary.json','representative_audio.csv','coverage_verification.json','final_audit_summary.json']
(Q/'four_class_freeze.json').write_text(json.dumps(dict(version='humbug-four-class-v1',release='0.0.1',zenodo=4904800,frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='GO_FOUR_CLASS_TRAINING',group_key='name',candidate_sha256=sha(Q/'four_class_split_candidate_metadata.csv'),metadata_sha256=sha(R/'sources/humbug_neurips_2021_zenodo_0_0_1.csv'),sha256={f:sha(Q/f) for f in files},test_policy='No test-driven selection, tuning, calibration or representative inspection. Preserve source membership through preprocessing and windowing.'),indent=2))
print(json.dumps(stats,indent=2))
