import sys,pathlib
sys.path.insert(0,r'D:\VGU-27\Edge AI Project\review_support\python_libs')
import fitz
p=fitz.open(r'C:\Users\minhl\Downloads\Synthetic_Swarm_Mosquito_Dataset_for_Acoustic_Classification__A_Proof_of_Concept (1).pdf')
for i in [2,3,4]:
    p[i].get_pixmap(matrix=fitz.Matrix(1.7,1.7)).save(r'D:\VGU-27\Edge AI Project\review_support\prior_page_'+str(i+1)+'.png')
