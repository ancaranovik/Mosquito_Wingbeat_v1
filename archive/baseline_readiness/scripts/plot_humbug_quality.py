from audit_humbug_core import R,Q,write
assert not (Q/'core_freeze.json').exists(),'Core-v1 is frozen; representative pre-freeze QC is already saved'
import json,csv
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
qc=json.loads((Q/'audio_qc.json').read_text())
core={r['id']:r['partition'] for r in csv.DictReader((Q/'core_split_candidate.csv').open())}
old={r['id']:r['candidate_partition'] for r in csv.DictReader((R/'manifests/humbug_metadata_candidate.csv').open())}
fig,axes=plt.subplots(3,3,figsize=(16,11),constrained_layout=True);chosen=[]
for row,sp in enumerate(['ae aegypti','an arabiensis','an funestus ss']):
 eligible=[r for r in qc.values() if r['species']==sp and (old[r['id']]=='training' if row==0 else core[r['id']]=='training')]
 eligible.sort(key=lambda r:(r['median_peak_to_median_db'],r['id']))
 for col,quant in enumerate([.1,.5,.9]):
  r=eligible[round(quant*(len(eligible)-1))];p=R/r['local_path'];y,sr=sf.read(p);y=y[:min(len(y),sr*8)];z=signal.resample_poly(y,80,441)
  f,t,S=signal.spectrogram(z,8000,nperseg=512,noverlap=448);db=10*np.log10(S+1e-20);db-=db.max()
  ax=axes[row,col];ax.pcolormesh(t,f,db,shading='auto',cmap='magma',vmin=-60,vmax=0);ax.set_ylim(0,2000)
  ax.set_title(f"{sp} | {r['id']} | QC percentile {int(quant*100)}\nRMS {r['rms_dbfs']:.1f} dBFS, spectral contrast {r['median_peak_to_median_db']:.1f} dB",fontsize=10)
  ax.set_xlabel('Seconds');ax.set_ylabel('Hz')
  chosen.append(dict(id=r['id'],species=sp,name=r['name'],local_path=r['local_path'],selection='development only; '+str(quant)+' quantile of spectral-contrast QC proxy',seconds_displayed=len(y)/sr,partition='training' if row else 'excluded_core_aegypti_previous_candidate_training'))
fig.suptitle('HumBugDB development audio QC: first up to 8 s; per-panel relative PSD, -60 to 0 dB\nSpectral ridges are compatible with tonal flight activity, not independent confirmation of species or mosquito presence.',fontsize=12)
fig.savefig(Q/'representative_spectrograms.png',dpi=140);write(Q/'representative_audio.csv',chosen)
print(json.dumps(chosen,indent=2))
