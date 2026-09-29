import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent / 'python_libs'))
import fitz
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('input_pdf', type=pathlib.Path)
p=fitz.open(parser.parse_args().input_pdf)
for i in [2,3,4]:
    p[i].get_pixmap(matrix=fitz.Matrix(1.7,1.7)).save(str(pathlib.Path(__file__).resolve().parent / ('prior_page_'+str(i+1)+'.png')))
