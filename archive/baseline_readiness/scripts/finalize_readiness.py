"""Prepare metadata and HOLD upload inventory only. Never train or inspect frozen audio."""
from pathlib import Path
import csv, json, hashlib, collections, zipfile
R=Path(__file__).resolve().parents[1]
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

# Preserve the existing Tiny MosquitoSong assignments, including held-out membership.
freeze=json.loads((R/'manifests/freeze.json').read_text())
for name,digest in freeze['manifest_sha256'].items():
    assert sha(R/'manifests'/name)==digest,name
tiny=read(R/'manifests/split_manifest.csv')
seen={};dedup=[]
for row in sorted(tiny,key=lambda r:r['source_identifier']):
    digest=row['pcm_sha256']
    if digest in seen:
        dedup.append(dict(source_identifier=row['source_identifier'],filename=row['filename'],canonical_source_identifier=seen[digest],exclusion_reason='exact PCM duplicate; omit from any later upload; assignment unchanged'))
    else:seen[digest]=row['source_identifier']
write(R/'manifests/duplicate_exclusions.csv',dedup)

# Metadata-only fallback candidate; selection is based on provenance, never outcomes.
allrows=read(R/'sources/humbug_neurips_2021_zenodo_0_0_1.csv')
cup=[r for r in allrows if r['country']=='Tanzania' and r['location_type']=='cup' and r['sound_type']=='mosquito']
chosen={'ae aegypti':'aedes_aegypti','an arabiensis':'anopheles_arabiensis','an funestus ss':'anopheles_funestus_ss'}
groups=collections.defaultdict(list)
for r in cup:groups[r['name']].append(r)
assert all(len({x['species'] for x in rr})==1 for rr in groups.values())
assign={};summary={}
for sp,label in chosen.items():
    names=sorted([n for n,rr in groups.items() if rr[0]['species']==sp],key=lambda n:hashlib.sha256(('20260926|'+sp+'|'+n).encode()).hexdigest())
    n=len(names); counts=[int(n*.6),int(n*.2),int(n*.2)]
    fractions=[n*.6-counts[0],n*.2-counts[1],n*.2-counts[2]]
    for i in sorted(range(3),key=lambda i:(-fractions[i],i))[:n-sum(counts)]:counts[i]+=1
    a,b,c=counts
    for p,nn in [('training',names[:a]),('validation',names[a:a+b]),('testing',names[a+b:])]:
        for name in nn:assign[name]=p
    rr=[x for x in cup if x['species']==sp]
    summary[sp]={'groups':n,'annotation_files':len(rr),'annotated_seconds':sum(float(x['length']) for x in rr),'group_split':dict(zip(['training','validation','testing'],counts)),'metadata_sample_rates':dict(collections.Counter(x['sample_rate'] for x in rr)),'gender':dict(collections.Counter(x['gender'] for x in rr)),'devices':dict(collections.Counter(x['device_type'] for x in rr))}
out=[]
for r in allrows:
    selected=r in cup and r['name'] in assign and r['species'] in chosen
    out.append(dict(**r,source_identifier=r['id']+'.wav',independent_group_identifier=('Tanzania/cup/'+r['name']) if selected else '',candidate_partition=assign[r['name']] if selected else 'excluded',label=chosen.get(r['species'],'') if selected else '',audio_audit_status='NOT_DOWNLOADED_NOT_VERIFIED',exclusion_reason='HOLD: audio integrity/duplicate/header audit pending' if selected else 'outside selected Tanzania cup three-species cohort'))
write(R/'manifests/humbug_metadata_candidate.csv',out)
(R/'humbug_metadata_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')

# Extract known archive members under safe stable names, with explicit labels; no split invented.
historical=read(R/'manifests/historical_archive_inventory.csv')
labels={'aedes_aegypti.zip':'aegypti','aedes_albopictus.zip':'albopictus','noise.zip':'noise','other_mosquitos_selected.zip':'other'}
uploads=[]
for row in historical:
    label=labels[row['archive']]; name=label+'_'+row['sha256'][:16]+'.wav'
    dest=R/'reference_upload_HOLD'/label/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        with zipfile.ZipFile(R/'sources/historical_archives'/row['archive']) as z:dest.write_bytes(z.read(row['path']))
    assert sha(dest)==row['sha256']
    uploads.append(dict(**row,label=label,local_path=dest.relative_to(R).as_posix(),source_identifier=row['archive']+'::'+row['path'],independent_group_identifier=label+'::'+row['recovered_filename_ancestor'],partition='HOLD_UNASSIGNED',exclusion_reason='no defensible three-way source split for original task; filename ancestors not verified individuals'))
write(R/'manifests/reference_upload_HOLD.csv',uploads)
newfiles=['manifests/humbug_metadata_candidate.csv','humbug_metadata_summary.json','manifests/reference_upload_HOLD.csv','manifests/duplicate_exclusions.csv']
(R/'manifests/readiness_freeze.json').write_text(json.dumps({'completed_date':'2026-09-26','tiny_original_freeze_date_note':'2026-09-23 was the initiation/seed date; audit completed 2026-09-26; original manifest retained unchanged','status':'NO_GO','humbug_seed':'20260926','humbug_status':'METADATA_ONLY_CANDIDATE_NO_AUDIO_AUDIT','sha256':{p:sha(R/p) for p in newfiles}},indent=2),encoding='utf8')
print(json.dumps({'humbug':summary,'reference_files_prepared':len(uploads),'duplicate_files_to_exclude':len(dedup)},indent=2))

