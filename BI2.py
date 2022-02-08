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
from skimage.io import imread,imsave,imshow,show
from skimage import exposure,draw
#from skimage.filters import threshold_triangle as tri
#from skimage.filters import threshold_yen as yen
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
        #show()
    return(df_img)

def mask_set(image_arr,set=''):
    if set == 'tr1':
        mask = np.ones(shape=image_arr.shape[0:2], dtype="bool")
        rr, cc = draw.disk((500, 500), 460)
        mask[rr, cc] = False
        image_arr[mask] = 0
        return(plt.imshow(image_arr))
    elif set == 'tr2':
        mask = np.ones(shape=image_arr.shape[0:2], dtype="bool")
        #rr, cc = draw.polygon((560,328,766),(25,117,178))
        rr, cc = draw.ellipse(455,168,80,180,rotation=np.deg2rad(45)) # better than polygon
        mask[rr, cc] = False
        rr1, cc1 = draw.disk((500, 500), 420)
        mask[rr1,cc1] =False
        image_arr[mask] = 0
        return(plt.imshow(image_arr))
    elif set == 'ch1':
        mask = np.ones(shape=image_arr.shape[0:2], dtype="bool")
        rr, cc = draw.disk((510, 505), 465)
        mask[rr, cc] = False
        image_arr[mask] = 0
        return(plt.imshow(image_arr))
    elif set == 'ch2':
        mask = np.ones(shape=image_arr.shape[0:2], dtype="bool")
        rr, cc = draw.disk((505, 500), 470)
        mask[rr, cc] = False
        image_arr[mask] = 0
        return(plt.imshow(image_arr))
    else:
        print('Specify set (tr1, tr2, ch1, or ch2).')

##############################
## Procedure #################
##############################

path = ('/home/howset/workspace/Bioimage2/Images/training_BF-C2DL-HSC/02/subset/*.tif') # E14 path
path = ('/home/howset/workspace/Bioimage2/Images/challenge_BF-C2DL-HSC/01/subset/*.tif')

path = ('/home/howsetya/workspace/Bioimage2/Images/training_BF-C2DL-HSC/01/subset/*.tif')
path = ('/home/howsetya/workspace/Bioimage2/Images/training_BF-C2DL-HSC/01/subset2/*.tif') #like above but more
path = ('/home/howsetya/workspace/Bioimage2/Images/training_BF-C2DL-HSC/02/subset/*.tif')
path = ('/home/howsetya/workspace/Bioimage2/Images/challenge_BF-C2DL-HSC/01/subset/*.tif')
path = ('/home/howsetya/workspace/Bioimage2/Images/challenge_BF-C2DL-HSC/02/subset/*.tif')

imgs = load_imgs(path)

#masking
for n in range(1,len(imgs),30):
    image_arr = imgs.iloc[n,1]
    mask_set(image_arr,'ch1')
    show()

# check
for n in range(1, len(imgs),5):
    #print(n)
    #a = imgs.iloc[n,1]
    #show()
    #image_arr = imgs.iloc[n,1]
    mask_set(image_arr)
    #show()
    

##############################
## tests #####################
##############################
imgs.iloc[0]
image=imgs.iloc[21,1]
imshow(image)
len(imgs)

#################################################
# edge based segmentation
# canny edge detection
from skimage import feature
from scipy import ndimage as ndi    
for n in range(len(imgs)):
    image = imgs.iloc[n,1]
    hist, hist_centers = exposure.histogram(image)
    title = imgs.iloc[n,0]
    print(title)
    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(image,sigma=1.6)
    #imsave(edges1)
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
    

filling = ndi.binary_fill_holes(edges1)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(filling, cmap=plt.cm.gray)
ax.set_title('filling the holes')
ax.axis('off')
#################################################
# region based segmentation
# sobel gradient
from skimage.filters import sobel

for n in range(len(imgs)):
    image = imgs.iloc[n,1]
    hist, hist_centers = exposure.histogram(image)
    title = imgs.iloc[n,0]
    print(title)
    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(image,sigma=1.6)
    #imsave(edges1)
    hist1, hist_centers1 = exposure.histogram(edges1)
    # Sobel gradient
    elmap = sobel(image)
    hist2, hist_centers2 = exposure.histogram(elmap)
    # display results
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    ax=ax.flatten()
    #ax[0].imshow(image, cmap='gray')
    #ax[0].set_title('original', fontsize=20)
    ax[0].imshow(edges1, cmap='gray')
    ax[0].set_title(r'Canny, $\sigma=1.6$', fontsize=20)
    ax[1].imshow(elmap,cmap='gray')
    ax[1].set_title('Sobel grad', fontsize=20)
    #ax[3].plot(hist)
    ax[2].plot(hist1)
    ax[3].plot(hist2)
    #for a in ax:
    #    a.axis('off')
    fig.tight_layout()
    plt.show()

elevation_map = sobel(image)
plt.imshow(elevation_map, cmap='gray')

markers = np.zeros_like(image)
markers[image < 30] = 1
markers[image > 150] = 2

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(markers, cmap=plt.cm.nipy_spectral)
ax.set_title('markers')
ax.axis('off')

from skimage import segmentation

segmentation_c = segmentation.watershed(elevation_map, markers)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(segmentation_c, cmap=plt.cm.gray)
ax.set_title('segmentation')
ax.axis('off')
#################################################
# labelling
from skimage.color import label2rgb

segmentation_c = ndi.binary_fill_holes(segmentation_c - 1)
labeled_c, _ = ndi.label(segmentation_c)
image_label_overlay = label2rgb(labeled_c, image=image, bg_label=0)

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes[0].imshow(image, cmap=plt.cm.gray)
axes[0].contour(segmentation_c, [0.5], linewidths=1.2, colors='y')
axes[1].imshow(image_label_overlay)

for a in axes:
    a.axis('off')

plt.tight_layout()

plt.show()
#################################################
#count colonies
import skimage
import matplotlib.pyplot as plt

def count_colonies(image_filename):
    #bacteria_image = skimage.io.imread(image_filename)
    #gray_bacteria = skimage.color.rgb2gray(bacteria_image)
    blurred_image = skimage.filters.gaussian(image_filename, sigma=1.0)
    mask = blurred_image < 0.2
    labeled_image, count = skimage.measure.label(mask, return_num=True)
    print(f"There are {count} colonies in {image_filename}")
    
    colored_label_image = skimage.color.label2rgb(labeled_image, bg_label=0)
    summary_image = skimage.color.gray2rgb(image_filename)
    summary_image[mask] = colored_label_image[mask]
    fig, ax = plt.subplots()
    plt.imshow(summary_image)

count_colonies(image)

#################################################
#masking
# def mask_tr1(image_arr):
#     mask = np.ones(shape=image_arr.shape[0:2], dtype="bool")
#     rr, cc = draw.disk((500, 500), 460)
#     mask[rr, cc] = False
#     image_arr[mask] = 0
#     return(plt.imshow(image_arr))

# def mask_tr2(image_arr):
#     mask = np.ones(shape=image_arr.shape[0:2], dtype="bool")
#     #rr, cc = draw.polygon((560,328,766),(25,117,178))
#     rr, cc = draw.ellipse(455,168,80,180,rotation=np.deg2rad(45)) # better than polygon
#     mask[rr, cc] = False
#     rr1, cc1 = draw.disk((500, 500), 420)
#     mask[rr1,cc1] =False
#     image_arr[mask] = 0
#     return(plt.imshow(image_arr))

# def mask_ch1(image_arr):
#     mask = np.ones(shape=image_arr.shape[0:2], dtype="bool")
#     rr, cc = draw.disk((510, 505), 465)
#     mask[rr, cc] = False
#     image_arr[mask] = 0
#     return(plt.imshow(image_arr))

# def mask_ch2(image_arr):
#     mask = np.ones(shape=image_arr.shape[0:2], dtype="bool")
#     rr, cc = draw.disk((505, 500), 470)
#     mask[rr, cc] = False
#     image_arr[mask] = 0
#     return(plt.imshow(image_arr))
#################################################
#################################################
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
#https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123 
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
    
import matplotlib.pyplot as plt
image = imgs.iloc[0,1]
thresh = tri(image)
binary = image > thresh
plt.imshow(binary,cmap='gray')

mask = edges1 < 0.2
fig, ax = plt.subplots()
plt.imshow(mask, cmap='gray')
plt.show()
import skimage
labeled_image, count = skimage.measure.label(mask, return_num=True)
print(count)

blurred_image = skimage.filters.gaussian(image, sigma=1.0)
