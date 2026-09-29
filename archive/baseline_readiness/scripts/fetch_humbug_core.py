from pathlib import Path
import sys,json,csv,io,zipfile,struct,zlib,hashlib,concurrent.futures,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'python_libs'))
import requests
rows=list(csv.DictReader((R/'sources/humbug_neurips_2021_zenodo_0_0_1.csv').open(encoding='utf-8-sig')))
selected={r['id'] for r in rows if r['country']=='Tanzania' and r['location_type']=='cup' and r['sound_type']=='mosquito' and r['species'] in ['ae aegypti','an arabiensis','an funestus ss']}
out=R/'raw/humbug_core';out.mkdir(parents=True,exist_ok=True)
def get(url,a,b):
 for attempt in range(6):
  try:
   q=requests.get(url,headers={'Range':f'bytes={a}-{b}'},timeout=90)
   q.raise_for_status();assert q.status_code==206 and len(q.content)==b-a+1,(q.status_code,len(q.content),a,b)
   return q.content
  except Exception:
   if attempt==5:raise
   time.sleep(2+attempt*2)
class Remote(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0
 def seek(self,off,whence=0):self.pos=off if whence==0 else self.pos+off if whence==1 else self.size+off;return self.pos
 def tell(self):return self.pos
 def seekable(self):return True
 def read(self,n=-1):
  n=min(n if n>=0 else self.size,self.size-self.pos)
  if n<=0:return b''
  a=self.pos;self.pos+=n;return get(self.url,a,a+n-1)
def archive(f):
 url=f['links']['self'];rr=Remote(url,f['size'])
 with zipfile.ZipFile(rr) as z: infos=[i for i in z.infolist() if Path(i.filename).stem in selected and i.filename.lower().endswith('.wav')]
 inventory=[dict(archive=f['key'],path=i.filename,offset=i.header_offset,compressed_bytes=i.compress_size,bytes=i.file_size,crc32=i.CRC) for i in infos]
 pending=[i for i in sorted(infos,key=lambda i:i.header_offset) if not (out/Path(i.filename).name).exists()]
 batches=[]
 for i in pending:
  end=i.header_offset+30+len(i.filename.encode())+65536+i.compress_size
  end=min(end,f['size']-1)
  if batches and i.header_offset-batches[-1][1]<65536 and end-batches[-1][0]<8*1024*1024:
   batches[-1][1]=max(end,batches[-1][1]);batches[-1][2].append(i)
  else:batches.append([i.header_offset,end,[i]])
 print(f['key'],'selected',len(infos),'range batches',len(batches),flush=True)
 def batch(job):
  k,(a,b,items)=job
  blob=get(url,a,b)
  for i in items:
   p=i.header_offset-a;h=struct.unpack_from('<4s5H3I2H',blob,p);assert h[0]==b'PK\x03\x04'
   start=p+30+h[-2]+h[-1];compressed=blob[start:start+i.compress_size]
   data=zlib.decompress(compressed,-15) if i.compress_type==8 else compressed
   assert len(data)==i.file_size and zlib.crc32(data)==i.CRC
   dest=out/Path(i.filename).name;temp=dest.with_suffix('.partial');temp.write_bytes(data);temp.replace(dest)
  if k%10==0:print(f['key'],k+1,'/',len(batches),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(batch,enumerate(batches)))
 return inventory
if __name__=='__main__':
 files=[f for f in json.loads((R/'sources/humbug_zenodo.json').read_text())['files'] if f['key'].endswith('.zip')]
 inventory=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  for result in ex.map(archive,files):inventory.extend(result)
 (R/'sources/humbug_selected_archive_inventory.json').write_text(json.dumps(inventory,indent=2))
 missing=selected-{p.stem for p in out.glob('*.wav')};print('Downloaded',len(selected)-len(missing),'missing',missing,flush=True)
