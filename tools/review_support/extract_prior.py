import sys, pathlib, json, hashlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'python_libs'))
from pypdf import PdfReader
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('input_directory', type=pathlib.Path)
src=parser.parse_args().input_directory
out=pathlib.Path(__file__).resolve().parent
for i,p in enumerate(sorted(src.glob('Synthetic_Swarm_Mosquito*.pdf'))):
    r=PdfReader(p)
    print(p.name, len(r.pages), dict(r.metadata))
    text='\n\n'.join('=== PDF PAGE '+str(j+1)+' ===\n'+page.extract_text() for j,page in enumerate(r.pages))
    target=out / ('prior_paper_'+str(i+1)+'.txt')
    target.write_text(text, encoding='utf-8')
    print('Extracted:', target, len(text))
