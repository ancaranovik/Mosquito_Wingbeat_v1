"""Deterministic metadata-only date-group split. No audio-derived split optimization."""
from audit_humbug_core import R,Q,rows,write
import collections,json,hashlib,random
import numpy as np
assert not (Q/'core_freeze.json').exists(),'Core-v1 is frozen; do not regenerate assignments in place'
species=['an arabiensis','an funestus ss']
byname=collections.defaultdict(list)
for r in rows:byname[r['name']].append(r)
eligible={n:v for n,v in byname.items() if v[0]['species'] in species and all(x['gender']=='Female' and x['fed']=='f' and x['method']=='LT' and x['plurality']=='Single' for x in v)}
dates=collections.defaultdict(set)
for n,v in eligible.items():dates[v[0]['record_datetime']].add(v[0]['species'])
shared=sorted(d for d,s in dates.items() if s==set(species))
counts=np.array([[sum(v[0]['record_datetime']==d and v[0]['species']==sp for v in eligible.values()) for sp in species] for d in shared])
assert len(shared)>=15
# Fixed candidate search budget and objective depend only on group counts.
rng=random.Random(20260926);best=None;nt=round(len(shared)*.6);nv=(len(shared)-nt)//2
for trial in range(20000):
 perm=list(range(len(shared)));rng.shuffle(perm);parts=[perm[:nt],perm[nt:nt+nv],perm[nt+nv:]]
 c=np.array([counts[p].sum(axis=0) for p in parts]);ratios=c/counts.sum(axis=0)
 score=float(np.sum((ratios-np.array([.6,.2,.2])[:,None])**2))
 candidate=(score,tuple(tuple(sorted(p)) for p in parts))
 if best is None or candidate<best:best=candidate
assignment={shared[i]:part for part,indices in zip(['training','validation','testing'],best[1]) for i in indices}
result=[]
for r in rows:
 reason=''
 if r['species'] not in species:reason='aegypti excluded: capture-method confound (LC only) and four dates'
 elif r['name'] not in eligible:reason='outside prespecified Female / fed=f / LT / Single cohort'
 elif r['record_datetime'] not in shared:reason='date has only one eligible species; exclude date confound'
 result.append(dict(**r,source_identifier=r['id']+'.wav',original_individual_group=r['name'],independent_group_identifier='Tanzania|Ifakara|cup|'+r['record_datetime'],partition='excluded' if reason else assignment[r['record_datetime']],exclusion_reason=reason,label={'an arabiensis':'anopheles_arabiensis','an funestus ss':'anopheles_funestus_ss'}.get(r['species'],''),local_path='raw/humbug_core/'+r['id']+'.wav',status='CANDIDATE_PENDING_COMPLETE_AUDIO_AUDIT'))
write(Q/'core_split_candidate.csv',result)
summary={}
for sp in species:
 summary[sp]={p:dict(individuals=len({r['name'] for r in result if r['species']==sp and r['partition']==p}),files=sum(r['species']==sp and r['partition']==p for r in result),dates=len({r['record_datetime'] for r in result if r['species']==sp and r['partition']==p}),seconds=sum(float(r['length']) for r in result if r['species']==sp and r['partition']==p)) for p in ['training','validation','testing']}
(Q/'split_design.json').write_text(json.dumps(dict(seed=20260926,search_candidates=20000,objective='squared error from 60/20/20 individual proportions per species, metadata only',shared_dates=len(shared),partition_dates=assignment,counts=summary),indent=2))
print(json.dumps(summary,indent=2))
