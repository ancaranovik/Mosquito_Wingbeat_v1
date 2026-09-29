import sys, pathlib, json, hashlib
sys.path.insert(0, r'D:\VGU-27\Edge AI Project\review_support\python_libs')
from pypdf import PdfReader
src=pathlib.Path(r'C:\Users\minhl\Downloads')
out=pathlib.Path(r'D:\VGU-27\Edge AI Project\review_support')
for i,p in enumerate(sorted(src.glob('Synthetic_Swarm_Mosquito*.pdf'))):
    r=PdfReader(p)
    print(p.name, len(r.pages), dict(r.metadata))
    text='\n\n'.join('=== PDF PAGE '+str(j+1)+' ===\n'+page.extract_text() for j,page in enumerate(r.pages))
    target=out / ('prior_paper_'+str(i+1)+'.txt')
    target.write_text(text, encoding='utf-8')
    print('Extracted:', target, len(text))
