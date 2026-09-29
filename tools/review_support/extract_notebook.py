import json,pathlib
p=pathlib.Path(__file__).resolve().parent / 'mosquito_swarm.ipynb'
n=json.loads(p.read_text(encoding='utf-8'))
out='\n\n'.join('=== CELL '+str(i)+' '+c['cell_type']+' ===\n'+''.join(c['source']) for i,c in enumerate(n['cells']))
p.with_suffix('.txt').write_text(out,encoding='utf-8')
print(len(n['cells']), len(out))
