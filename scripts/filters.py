import os
import sys
import shutil
import cv2
import random
import argparse
from tqdm import tqdm
from PIL import Image

def getparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-filts', '--filters', type=str)
    parser.add_argument('-dset_p', '--path_dset', type=str)
    parser.add_argument('-filters', '--filters_to_apply', type=list, default=['QF30', 'QF50', 'QF70', 'QF90'])
    parser.add_argument('-perc_take', '--perc_to_take', type=float, default=1)

    args = parser.parse_args()
    return args

def JPEGCompr(image_path, path_test, image, fold, qfac = ['30', '50', '70', '90']):
    for q in qfac:
        try:
            im = Image.open(image_path)
            im.save(path_test+"QF"+str(q)+"/"+fold+"/"+image[:-4]+'.jpg',format='jpeg', subsampling=0, quality=int(q))
        except:
            print("**Errore nell'elaborazione della foto: ", image_path)

def main():
    parser = getparser()
    path_dset = parser.pathdset
    perc_to_take = parser.perc_to_take
    filters = parser.filters_to_apply
    path_test = path_dset + '-jpeg'
    for fil in filters:
        if not os.path.exists(path_test+fil):
            os.makedirs(path_test+fil)
        for architecture in os.listdir(path_dset):
            architecture_path = os.path.join(path_dset, architecture)
            for fold in os.listdir(architecture_path):
                if not os.path.exists(os.path.join(path_test+fil, architecture, fold)):
                    os.makedirs(os.path.join(path_test+fil, architecture, fold))

    for architecture in os.listdir(path_dset):
        architecture_path = os.path.join(path_dset, architecture)
        for model in os.listdir(architecture_path):
            model_path = os.path.join(architecture_path, model)
            num = int(len(os.listdir(model_path))*perc_to_take)
            imgs_sample = random.sample(os.listdir(model_path), num)
            for img in tqdm(imgs_sample, desc=f'{model}'):
                img_path = os.path.join(model_path, img)        
                JPEGCompr(img_path, path_test, img, os.path.join(architecture, model))

    for q in filters:
        print(f'\n- {q}:')
        for architecture in os.listdir(path_test+q):
            architecture_path = os.path.join(path_dset, architecture)
            for model in os.listdir(architecture_path):
                model_path = os.path.join(architecture_path, model)
                print(f'number of compressed in {q} images for {model}: {len(os.listdir(os.path.join(path_test+q,architecture, model)))}')  

if __name__=='__main__':
    main()