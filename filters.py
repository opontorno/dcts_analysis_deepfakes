import os
import sys
import shutil
import cv2
from tqdm import tqdm
from PIL import Image
import random

def Rotation(img, name, pathOut, fold):
    try:
        h, w, _= img.shape
        center = (w / 2, h / 2)
        rotation = [45, 135, 225, 315]
        for r in rotation:
            
            M = cv2.getRotationMatrix2D(center, r, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(pathOut+"rot-"+str(r)+"/"+fold+"/"+name.split('.')[0]+".png", rotated)
    except:
        print("**Errore nell'elaborazione della foto - Rotazione ")

def Mirror(img, name, pathOut, fold):
    try:
        flipBoth = cv2.flip(img, -1)
        cv2.imwrite(pathOut+"mir-B/"+fold+"/"+name.split('.')[0]+".png", flipBoth)
    except:
        print("**Errore nell'elaborazione della foto - Mirror ")
        
def Scaling(img, name, pathOut, fold):
    
    try:
        #ZOOM- 50% in meno
        scale_percent = 50 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        cv2.imwrite(pathOut+"scal-+50/"+fold+"/"+name.split('.')[0]+".png", resized)
    except:
        print("**Errore nell'elaborazione della foto - Zoom- ")
        
    try:
        scale_percent = 200 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
    
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(pathOut+"scal-50/"+fold+"/"+name.split('.')[0]+".png", resized)
    except:
        print("**Errore nell'elaborazione della foto - Zoom+ ")
    
def GaussNoise(img, name, pathOut, fold):

    kernel = [3,9]#,15]
    for k in kernel:
        try:
            dst = cv2.GaussianBlur(img,(k,k),cv2.BORDER_DEFAULT)
            cv2.imwrite(pathOut+"GaussNoise-"+str(k)+"/"+fold+"/"+name.split('.')[0]+".png", dst)
        except:
            print("**Errore nell'elaborazione della foto - Gaussian Noice ")

def JPEGCompr(image_path, path_test, image, fold):
    #qfac = ['1','10','20','30','40', '50', '60', '70', '80', '90']
    qfac = ['30', '50', '70', '90']
    for q in qfac:
        try:
            im = Image.open(image_path)
            im.save(path_test+"QF"+str(q)+"/"+fold+"/"+image[:-4]+'.jpg',format='jpeg', subsampling=0, quality=int(q))
        except:
            print("**Errore nell'elaborazione della foto: ", image_path)


#filters = ['GaussNoise-3', 'GaussNoise-9', 'GaussNoise-15', 'mir-B', 'QF-50', 'QF-60', 'QF-70', 'QF-80', 'QF-90', 'rot-45', 'rot-135', 'rot-225', 'rot-315', 'scal-+50', 'scal-50']

filters = ['QF30', 'QF50', 'QF70', 'QF90']

path_dset = "/home/opontorno/data/opontorno/datasets"
path_test = path_dset + '-jpeg'
perc_to_take = 1

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
            
            #-----------------------------------------------------------------------------------------------
            #       JPEG Compression
            #-----------------------------------------------------------------------------------------------
        
            JPEGCompr(img_path, path_test, img, os.path.join(architecture, model))
            
            # #-----------------------------------------------------------------------------------------------
            # #       Mirror
            # #-----------------------------------------------------------------------------------------------
            
            # #continue

            # im = cv2.imread(pathIn + fold + "/" + image)
            
            # Mirror(im, image, pathOut, fold)
                
            # #-----------------------------------------------------------------------------------------------
            # #       Rotation
            # #-----------------------------------------------------------------------------------------------
        
            # Rotation(im, image, pathOut, fold)
            
            # #-----------------------------------------------------------------------------------------------
            # #       Add Gaussian Blur
            # #-----------------------------------------------------------------------------------------------
        
            # GaussNoise(im, image, pathOut, fold)
                
            # #-----------------------------------------------------------------------------------------------
            # #       ZoomIn ZoomOut
            # #-----------------------------------------------------------------------------------------------
            
            # Scaling(im, image, pathOut, fold)
    

for q in filters:
    print(f'\n- {q}:')
    for architecture in os.listdir(path_test+q):
        architecture_path = os.path.join(path_dset, architecture)
        for model in os.listdir(architecture_path):
            model_path = os.path.join(architecture_path, model)
            print(f'number of compressed in {q} images for {model}: {len(os.listdir(os.path.join(path_test+q,architecture, model)))}')  