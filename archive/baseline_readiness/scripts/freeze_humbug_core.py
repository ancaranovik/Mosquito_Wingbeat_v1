"""Freeze the audited, controlled core cohort. No training or audio conversion."""
from audit_humbug_core import R,Q,write
import json,csv,collections,hashlib,datetime
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
freeze=Q/'core_freeze.json'
assert not freeze.exists(),'Existing final freeze must not be overwritten'
qc=json.loads((Q/'audio_qc.json').read_text());assert len(qc)==1301
overlap=json.loads((Q/'overlap_summary.json').read_text());assert overlap['cross_original']==0
assert len({x['pcm_sha256'] for x in qc.values()})==1301
candidate=list(csv.DictReader((Q/'core_split_candidate.csv').open()))
final=[]
for r in candidate:
 a=qc[r['id']]
 assert a['actual_duration']>=1 and a['zero_fraction']<1 and a['clip_fraction']==0
 final.append(dict(**{k:v for k,v in r.items() if k not in ['status','sample_rate']},metadata_sample_rate=r['sample_rate'],actual_duration=a['actual_duration'],sample_rate=a['actual_sample_rate'],channels=a['channels'],subtype=a['subtype'],file_sha256=a['file_sha256'],pcm_sha256=a['pcm_sha256'],status='EXCLUDED_CORE_V1' if r['partition']=='excluded' else 'FROZEN_CORE_V1'))
write(Q/'core_source_manifest.csv',final)
included=[r for r in final if r['partition']!='excluded'];assert len(included)==668
for key in ['name','independent_group_identifier','pcm_sha256']:
 groups=collections.defaultdict(set)
 for r in included:groups[r[key]].add(r['partition'])
 assert all(len(x)==1 for x in groups.values()),key
group_rows=[]
for n in sorted({r['name'] for r in included}):
 rr=[r for r in included if r['name']==n];a=rr[0]
 group_rows.append(dict(original_individual_group=n,species=a['species'],date_group=a['independent_group_identifier'],partition=a['partition'],files=len(rr),seconds=sum(float(r['actual_duration']) for r in rr)))
write(Q/'core_group_manifest.csv',group_rows)
for p in ['training','validation','testing']:write(Q/('core_'+p+'_files.csv'),[r for r in included if r['partition']==p])
summary={}
for sp in sorted({r['species'] for r in final}):
 rr=[x for x in qc.values() if x['species']==sp]
 summary[sp]=dict(files=len(rr),original_groups=len({x['name'] for x in rr}),recording_dates=len({x['record_datetime'] for x in rr}),seconds=sum(x['actual_duration'] for x in rr),duration_range=[min(x['actual_duration'] for x in rr),max(x['actual_duration'] for x in rr)],rms_dbfs_range=[min(x['rms_dbfs'] for x in rr),max(x['rms_dbfs'] for x in rr)],rms_dbfs_median=float(np.median([x['rms_dbfs'] for x in rr])),tonality_proxy_median=float(np.median([x['tonal_frame_fraction_10db'] for x in rr])))
(Q/'audit_summary.json').write_text(json.dumps(dict(species=summary,format='44100 Hz mono PCM24 WAV',max_duration_error_seconds=max(abs(x['duration_error_seconds']) for x in qc.values()),exact_duplicate_pairs=0,overlap=overlap,clipped_files=0,digital_silence_files=0,core_files=len(included),core_original_groups=len(group_rows),core_date_groups=len({r['independent_group_identifier'] for r in included}),core_seconds=sum(r['actual_duration'] for r in included)),indent=2))
files=['core_source_manifest.csv','core_group_manifest.csv','core_training_files.csv','core_validation_files.csv','core_testing_files.csv','split_design.json','audio_inventory.csv','exact_overlap.csv','overlap_summary.json','representative_audio.csv','audit_summary.json']
freeze.write_text(json.dumps(dict(version='humbug-core-v1',frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='CONDITIONAL_GO_CONTROLLED_TWO_SPECIES_CORE_ONLY',test_policy='No further listening, plots, feature tuning, predictions or selection on testing members until final locked evaluation; fixed preprocessing and integrity checks only.',supersedes='manifests/humbug_metadata_candidate.csv is a preserved pre-audio three-species proposal, never an approved final split',selection='Female; fed=f; method=LT; plurality=Single; Tanzania/Ifakara/cup; arabiensis or funestus ss; dates with both eligible species',grouping='All clips of each original name stay together; all selected original names on the same record_datetime stay together',sha256={f:sha(Q/f) for f in files},metadata_sha256=sha(R/'sources/humbug_neurips_2021_zenodo_0_0_1.csv'),archive_inventory_sha256=sha(R/'sources/humbug_selected_archive_inventory.json')),indent=2))
print((Q/'audit_summary.json').read_text())
