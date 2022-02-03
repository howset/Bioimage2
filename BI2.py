#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 13:53:58 2022

@author: howset
"""

##############################
## Import packages ###########
##############################

import glob, re
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread,imshow,show
from skimage import exposure
from skimage.filters import threshold_triangle as tri
from skimage.filters import threshold_yen as yen

##############################
## Functions #################
##############################

def load_imgs(path):
    files_img = glob.glob(path)
    files_img.sort()
    df_img = pd.DataFrame(columns=['FileName','Image'])
    for file in files_img:
        print(file) # to show progress
        filename = re.sub(".*/", "", file[:-4])
        img = imread(file) # tifffile plugin for loading compressed tiffs requires imagecodecs
        df_img.loc[len(df_img)] = [filename,img]
        #imshow(img)
        #show()
    return(df_img)


##############################
## Procedure #################
##############################

path = ('/home/howset/workspace/Bioimage2/Images/training_BF-C2DL-HSC/02/subset/*.tif') # E14 path
path = ('/home/howset/workspace/Bioimage2/Images/challenge_BF-C2DL-HSC/01/subset/*.tif')
imgs = load_imgs(path)


##############################
## tests #####################
##############################
imgs.iloc[0]
imgs.iloc[0,1]
imshow(imgs.iloc[0,1])
len(imgs)

for n in range(len(imgs)):
    image = imgs.iloc[n,1]
    imagemm = exposure.rescale_intensity(image)
    title = imgs.iloc[n,0]
    print(title)
    thresh = tri(image)
    binary = image > thresh
    #plt.imshow(binary,cmap='gray')
    plt.hist(imagemm)
    plt.title(title)
    show()


import matplotlib.pyplot as plt
image = imgs.iloc[0,1]
thresh = tri(image)
binary = image > thresh
plt.imshow(binary,cmap='gray')

fig, ax = plt.subplots(figsize=(5, 5))
qcs = ax.contour(image, origin='image')
ax.set_title('Contour plot of the same raw image')
plt.show()

from skimage.filters import try_all_threshold
img = imgs.iloc[20,1]
fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
plt.show()

from skimage.filters import try_all_threshold
for n in range(len(imgs)):
    img = imgs.iloc[n,1]
    title = imgs.iloc[n,0]
    fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
    plt.title(title)
    plt.show()
    
imagemm = exposure.rescale_intensity(image)
imshow(imagemm)

img = imgs.iloc[0,1]
p2, p98 = np.percentile(img, (2, 98))
imgmm =exposure.rescale_intensity(imgs.iloc[0,1])
p2, p98 = np.percentile(img, (2, 98))

plt.hist(img)
    
    
    
    
    