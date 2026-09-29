import sys,pathlib,zipfile,io,json,csv,collections,re,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python_libs'))
import soundfile as sf
rows=[]; archives=[]
for p in sorted((ROOT/'sources'/'historical_archives').glob('*.zip')):
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            if not n.lower().endswith('.wav') or n.startswith('__MACOSX/') or pathlib.PurePosixPath(n).name.startswith('._'):continue
            b=z.read(n); info=sf.info(io.BytesIO(b))
            rows.append(dict(archive=p.name,path=n,sample_rate=info.samplerate,channels=info.channels,subtype=info.subtype,duration_seconds=info.duration,sha256=hashlib.sha256(b).hexdigest(),recovered_filename_ancestor=re.sub(r'^16khz_\d+_','',pathlib.PurePosixPath(n).name)))
    rr=[r for r in rows if r['archive']==p.name]
    archives.append(dict(archive=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),files=len(rr),duration_seconds=sum(r['duration_seconds'] for r in rr),durations_min_max=[min(r['duration_seconds'] for r in rr),max(r['duration_seconds'] for r in rr)],formats=dict(collections.Counter(f"{r['sample_rate']}/{r['channels']}/{r['subtype']}" for r in rr)),filename_ancestor_count=len(set(r['recovered_filename_ancestor'] for r in rr))))
with (ROOT/'manifests'/'historical_archive_inventory.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'historical_archive_summary.json').write_text(json.dumps(archives,indent=2))
print(json.dumps(archives,indent=2))
