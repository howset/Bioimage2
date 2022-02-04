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
import numpy as np

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
path = ('/home/howsetya/workspace/Bioimage2/Images/training_BF-C2DL-HSC/01/subset/*.tif')
imgs = load_imgs(path)


##############################
## tests #####################
##############################
imgs.iloc[0]
imgs.iloc[0,1]
imshow(imgs.iloc[0,1])
len(imgs)


for n in range(len(imgs)):
    fig, axes = plt.subplots(1, 4, figsize=(10, 10))
    ax = axes.flatten()
    image = imgs.iloc[n,1]
    imagemm = exposure.rescale_intensity(image)
    title = imgs.iloc[n,0]
    print(title)
    hist_im, hist_centers_im = exposure.histogram(image)
    hist_imm, hist_centers_imm = exposure.histogram(imagemm)
    thresh = tri(image)
    binary = image > thresh
    ax[0].imshow(image)
    ax[1].imshow(plt.plot(hist_im))
    ax[2].imshow(imagemm)
    ax[3].imshow(plt.plot(hist_imm))
    #plt.imshow(binary,cmap='gray')
    #plt.hist(imagemm)
    #plt.title(title)
    show()

#https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
#https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_local_equalize.html
for n in range(len(imgs)):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax = axes.flatten()
    image = imgs.iloc[n,1]
    title = imgs.iloc[n,0]
    print(title)
    hist, hist_centers = exposure.histogram(image)
    imagemm = exposure.equalize_hist(image)
    histmm, hist_centersmm = exposure.histogram(imagemm)
    ax[0].imshow(image,cmap='gray')
    ax[1].plot(hist)
    ax[2].imshow(imagemm,cmap='gray')
    ax[3].plot(histmm)

#https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
#https://scikit-image.org/docs/dev/auto_examples/edges/plot_canny.html    
from skimage import feature
#from scipy import ndimage as ndi
for n in range(len(imgs)):
    image = imgs.iloc[n,1]
    title = imgs.iloc[n,0]
    print(title)
    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(image,sigma=2)
    edges2 = feature.canny(image, sigma=3)
    # display results
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('original', fontsize=20)
    ax[1].imshow(edges1, cmap='gray')
    ax[1].set_title(r'Canny filter, $\sigma=2$', fontsize=20)
    ax[2].imshow(edges2, cmap='gray')
    ax[2].set_title(r'Canny filter, $\sigma=3$', fontsize=20)
    for a in ax:
        a.axis('off')
    fig.tight_layout()
    plt.show()

#this one works    
for n in range(len(imgs)):
    image = imgs.iloc[n,1]
    hist, hist_centers = exposure.histogram(image)
    title = imgs.iloc[n,0]
    print(title)
    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(image,sigma=1.6)
    hist1, hist_centers1 = exposure.histogram(edges1)
    # display results
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    ax=ax.flatten()
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('original', fontsize=20)
    ax[1].imshow(edges1, cmap='gray')
    ax[1].set_title(r'Canny filter, $\sigma=1.5$', fontsize=20)
    ax[2].plot(hist)
    ax[3].plot(hist1)
    #for a in ax:
    #    a.axis('off')
    fig.tight_layout()
    plt.show()
    
import matplotlib.pyplot as plt
image = imgs.iloc[0,1]
thresh = tri(image)
binary = image > thresh
plt.imshow(binary,cmap='gray')

