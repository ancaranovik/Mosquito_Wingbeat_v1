from audit_humbug_four import Q
import csv,numpy as np,soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rr=list(csv.DictReader((Q/'representative_audio.csv').open()))
fig,axes=plt.subplots(4,3,figsize=(15,10),constrained_layout=True)
for ax,r in zip(axes.flat,rr):
 x,sr=sf.read(r['local_path']);x=x[:int(float(r['seconds_displayed'])*sr)]
 n=len(x)//441;x=x[:n*441].reshape(n,441);t=np.arange(n)*.01
 ax.fill_between(t,x.min(1),x.max(1),linewidth=.4)
 ax.set_title(r['species']+' / '+r['id']);ax.set_xlabel('Seconds');ax.set_ylabel('PCM amplitude')
fig.savefig(Q/'representative_waveforms.png',dpi=120)
