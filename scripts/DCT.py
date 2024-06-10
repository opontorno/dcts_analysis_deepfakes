from PIL import Image
import numpy as np
import scipy.stats as st
from scipy.fft import dct
from scipy import fftpack
import time
from tqdm import tqdm
import os
import statistics 
import math
import pickle
import argparse

def getparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dset_path', '--datasets_path', type=str)
    parser.add_argument('-dcts', '--dcts_path', type=str)
    parser.add_argument('-DC_coeff', '--includeDC_coeff', type=bool, default=True)
    parser.add_argument('laplacian_stats', '--get_laplacian_stats', type=bool, default=False)

    args = parser.parse_args()
    return args

def dct2(a):
    return fftpack.dct( fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def GetId(q):
    mask = [0,1,5,6,14,15,27,28, 2,4,7,13,16,26,29,42, 3,8,12,17,25,30,41,43, 9,11,18,24,31,40,44,53, 10,19,23,32,39,45,52,54, 20,22,33,38,46,51,55,60, 21,34,37,47,50,56,59,61, 35,36,48,49,57,58,62,63]
    return mask.index(q)

def image_to_blocks(image):
    blocks=[]
    dctblocks = []
    for i in range(0,len(image),8):
        for j in range(0,len(image[0]),8):
            blocks.append((np.array(image[i:(i+8),j:(j+8)])))
            dctblocks.append(dct2(np.array(image[i:(i+8),j:(j+8)])).reshape(-1))
    return blocks, np.array(dctblocks)

## ATTENTION! The following script assumes that images are organised hierarchically in two folder levels (example: Datasets -> GAN -> CycleGAN -> image).
# Start dct coefficients calculus
def main():
    parser = getparser()
    if not os.path.exists(parser.dcts_path):
        os.makedirs(parser.dcts_path)
    start_time = time.time()
    for model in os.listdir(parser.datasets_path):
        print(f'- {model}')
        architectures_path = os.path.join(parser.datasets_path, model)
        for architecture in os.listdir(architectures_path):
            architecture_path = os.path.join(architectures_path, architecture)     
            if not os.path.exists(os.path.join(parser.dcts_path, model, architecture)):
                os.makedirs(os.path.join(parser.dcts_path, model, architecture))
                
            for image in tqdm(os.listdir(architecture_path), desc=f'{architecture}'):
                if os.path.exists(os.path.join(parser.dcts_path, model, architecture, image[:-4])+'-DCAC.pkl'):
                    continue
                img_path = os.path.join(architecture_path,image)            
                try:
                    img = Image.open(img_path).convert('L')
                    
                    width, height = img.size
                    while True:
                        if width % 8 != 0:
                            width -= 1
                        if height % 8 != 0:
                            height -= 1
                        if width % 8 == 0 and height % 8 == 0:
                            break
                    im = np.asarray(img)
                    im = im[:height, :width]    
                    blocks, dctblocks = image_to_blocks(im)
                    lsDCAC = []
                    lsBETA = []
                    lsMIU = []
                    start = 1 if parser.includeDC_coeff == False else 0
                    for i in range(start, 64):
                        lsDCAC.append(statistics.stdev(dctblocks[:, GetId(i)])/math.sqrt(2))
                        if parser.get_laplacian_stats:
                            laplacian = st.laplace.fit(dctblocks[:, GetId(i)])
                            miu = laplacian[0]
                            beta = laplacian[1]
                            lsBETA.append(beta)
                            lsMIU.append(miu)
                    
                    with open(os.path.join(parser.dcts_path, model, architecture, image[:-4])+'-DCAC.pkl', 'wb') as f:
                        pickle.dump(lsDCAC, f)
                        
                    if parser.get_laplacian_stats:                    
                        with open(os.path.join(parser.dcts_path, model, architecture, image[:-4])+'-BETA.pkl', 'wb') as f:
                            pickle.dump(lsBETA, f)

                        with open(os.path.join(parser.dcts_path, model, architecture, image[:-4])+'-MIU.pkl', 'wb') as f:
                            pickle.dump(lsMIU, f)
                        
                except:
                    print("Errore nell'elaborazione dell'immagine ", img_path)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\ntotal esecution time: {(execution_time/60):.2f} minutes\n")

    ## Control 
    print('Control: ')
    for model in os.listdir(parser.datasets_path):
        architectures_path = os.path.join(parser.datasets_path, model)
        for architecture in os.listdir(architectures_path):
            images_path = os.listdir(os.path.join('/home/opontorno/data/opontorno/datasets', model, architecture))
            parser.dcts_path = os.listdir(os.path.join('dcts', model, architecture))
            print(f'OK - all {architecture} images were used' if len(images_path)==len(parser.dcts_path) else f'PROBLEM - {architecture}')

if __name__=='__main__':
    main()