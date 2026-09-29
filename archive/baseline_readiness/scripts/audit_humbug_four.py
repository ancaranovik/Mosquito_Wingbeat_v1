"""Audio integrity/QC only; no model training or class prediction."""
from pathlib import Path
import sys,csv,json,hashlib,collections
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'python_libs'))
import numpy as np
import soundfile as sf
from scipy import signal
Q=R/'humbug_four_audit';Q.mkdir(exist_ok=True)
rows=list(csv.DictReader((R/'sources/humbug_neurips_2021_zenodo_0_0_1.csv').open(encoding='utf-8-sig')))
rows=[r for r in rows if r['country']=='Tanzania' and r['location_type']=='cup' and r['sound_type']=='mosquito' and r['species'] in ['ae aegypti','an arabiensis','culex pipiens complex','ma uniformis']]
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,rr):
 with p.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def audit():
 cache=Q/'audio_qc.json';results=json.loads(cache.read_text()) if cache.exists() else {}
 for j,r in enumerate(rows):
  if r['id'] in results:continue
  p=R.parent/'data/audio'/(r['id']+'.wav')
  if not p.exists():continue
  info=sf.info(p);x,sr=sf.read(p,dtype='int32',always_2d=True);y=x.astype('float64')/2**31
  assert sr==44100 and x.shape[1]==1,(p,info)
  y=y[:,0];z=signal.resample_poly(y,80,441)
  f,t,S=signal.spectrogram(z,8000,nperseg=512,noverlap=256,mode='psd')
  band=(f>=150)&(f<=1200);b=S[band];contrast=10*np.log10((np.max(b,axis=0)+1e-20)/(np.median(b,axis=0)+1e-20))
  result=dict(**r,local_path=str(p.resolve()),actual_sample_rate=sr,channels=info.channels,subtype=info.subtype,actual_duration=info.duration,duration_error_seconds=info.duration-float(r['length']),file_sha256=sha(p.read_bytes()),pcm_sha256=sha(x.tobytes()),peak=float(np.max(np.abs(y))),rms_dbfs=float(20*np.log10(np.sqrt(np.mean(y*y))+1e-20)),dc=float(np.mean(y)),clip_fraction=float(np.mean(np.abs(y)>=.999)),zero_fraction=float(np.mean(y==0)),spectral_peak_hz=float(f[band][np.argmax(np.mean(b,axis=1))]),median_peak_to_median_db=float(np.median(contrast)),tonal_frame_fraction_10db=float(np.mean(contrast>10)))
  results[r['id']]=result
  if j%50==0:cache.write_text(json.dumps(results,indent=2));print('QC',len(results),'/',len(rows),flush=True)
 cache.write_text(json.dumps(results,indent=2));print('QC available',len(results),'expected',len(rows),flush=True)
 if len(results)==len(rows):write(Q/'audio_inventory.csv',[results[r['id']] for r in rows])
 return results
if __name__=='__main__':audit()

