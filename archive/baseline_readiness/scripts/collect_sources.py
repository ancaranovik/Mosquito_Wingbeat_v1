"""Download public source inventory/audio only. Never executes research notebooks."""
import sys, pathlib, json, dataclasses, concurrent.futures
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python_libs'))
import gdown, requests, soundfile as sf, io, time

def main():
    rows=[]
    cached=ROOT/'sources'/'drive_inventory.json'
    if cached.exists():
        rows=json.loads(cached.read_text())
    else:
        for kind,folder in [('mosquito','1AksZW-3KCSPQqsPJ8LwyS1i_qjVDvtI6'),('noise','12enzpCQlpVEAevcasrk0zs5Lu0_qjSIQ')]:
            files=gdown.download_folder(id=folder,output=str(ROOT/'raw'/kind),skip_download=True,use_cookies=False,quiet=True,timeout=60)
            print(kind,len(files),flush=True)
            for f in files:
                rows.append(dict(kind=kind,id=f.id,path=f.path,local_path=f.local_path))
    (ROOT/'sources'/'drive_inventory.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    if '--download' in sys.argv:
        def get(r):
            p=pathlib.Path(r['local_path']); p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists() and p.stat().st_size>44:
                try:
                    sf.read(p,dtype='int16')
                    return None
                except Exception:pass
            try:
                response=requests.get('https://drive.google.com/uc?id='+r['id'],timeout=60)
                response.raise_for_status()
                data=response.content
                if data[:4] != b'RIFF':raise ValueError('Public response was not WAV audio; download stopped for this file')
                sf.read(io.BytesIO(data),dtype='int16')
                temp=p.with_suffix('.download')
                temp.write_bytes(data)
                temp.replace(p)
            except Exception as e:return {'file':r['path'],'error':str(e)}
        errors=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for i,result in enumerate(ex.map(get,rows),1):
                if result:
                    errors.append(result)
                    if len(errors)<=3:print(result,flush=True)
                if i%50==0:print('downloaded/attempted',i,'errors',len(errors),flush=True)
        (ROOT/'sources'/'download_errors.json').write_text(json.dumps(errors,indent=2))
        print('complete',len(rows),'errors',len(errors),flush=True)
if __name__=='__main__':main()
