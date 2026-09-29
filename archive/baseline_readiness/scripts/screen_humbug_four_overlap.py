"""Exact-PCM overlap screen with three non-silent 100 ms anchors per clip.
The first 32 samples nominate a match; full 100 ms bytes must agree.
No claim is made about gain-scaled or noisy duplicate recordings.
"""
from audit_humbug_four import R,Q,rows,write
assert not (Q/'core_freeze.json').exists(),'Core-v1 is frozen; use saved pre-freeze overlap results'
import soundfile as sf
import numpy as np
import ahocorasick,collections,json
A=ahocorasick.Automaton();patterns=collections.defaultdict(list)
for r in rows:
 p=R.parent/'data/audio'/(r['id']+'.wav');x,sr=sf.read(p,dtype='int32');n=int(sr*.1)
 for pos in sorted({0,max(0,(len(x)-n)//2),max(0,len(x)-n)}):
  anchor=x[pos:pos+n]
  if len(anchor)!=n or np.std(anchor.astype(float))<2**15:continue
  raw=anchor.tobytes();patterns[raw[:128].decode('latin1')].append((r['id'],pos,raw))
for pat,items in patterns.items():A.add_word(pat,items)
A.make_automaton();print('anchor patterns',len(patterns),flush=True)
meta={r['id']:r for r in rows};matches={}
for j,r in enumerate(rows):
 x,sr=sf.read(R.parent/'data/audio'/(r['id']+'.wav'),dtype='int32');raw=x.tobytes()
 for end,anchors in A.iter(raw.decode('latin1')):
  offset=end-127
  if offset%4:continue
  for source,pos,anchor in anchors:
   if source==r['id'] or raw[offset:offset+len(anchor)]!=anchor:continue
   key=tuple(sorted((source,r['id'])))
   if key not in matches:
    a=meta[source];matches[key]=dict(source_a=source,source_b=r['id'],name_a=a['name'],name_b=r['name'],species_a=a['species'],species_b=r['species'],offset_a=pos,offset_b=offset//4,samples=len(anchor)//4,cross_original=a['name']!=r['name'],cross_date=a['record_datetime']!=r['record_datetime'])
 if j%100==0:print('overlap scanned',j,'matches',len(matches),flush=True)
results=list(matches.values())
if results:write(Q/'exact_overlap.csv',results)
else:(Q/'exact_overlap.csv').write_text('source_a,source_b,name_a,name_b,species_a,species_b,offset_a,offset_b,samples,cross_original,cross_date\n')
(Q/'overlap_summary.json').write_text(json.dumps(dict(files=len(rows),anchor_patterns=len(patterns),matches=len(results),cross_original=sum(x['cross_original'] for x in results),cross_date=sum(x['cross_date'] for x in results)),indent=2))
print((Q/'overlap_summary.json').read_text(),flush=True)

