#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:31:53 2022

@author: howsetya
"""

##############################
## Imports ###################
##############################

import sys, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import draw, util
from skimage.feature import canny
from skimage.morphology import binary_dilation, disk
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from scipy.optimize import curve_fit

##############################
## Functions #################
##############################
def usage():
    print('')
    print('Usage: python3 cell_count.py Req1 Req2 Opt3')
    print('Req1: path to images')
    print('Req2: image set (tr1, tr2, ch1, or ch2)')
    print('Opt3: plot individual image (non functional)')
    print('')

def load_imgs(path):
    files_img = glob.glob(f'{path}*.tif')
    files_img.sort()
    df_img = pd.DataFrame(columns=['FileName','Image'])
    for file in files_img:
        print(f'Loading {file}') 
        filename = re.sub(".*/", "", file[:-4])
        img = imread(file,plugin='tifffile') # tifffile plugin for loading compressed tiffs requires imagecodecs
        df_img.loc[len(df_img)] = [filename,img]
    return df_img

def mask_set(image_arr,set=''):
    image_arr_copy = np.squeeze(image_arr) #not sure why squeeze is necessary
    if len(image_arr.shape)>2:
        mask = np.ones(shape=image_arr.shape[1:3],dtype="bool")
    else:
        mask = np.ones(shape=image_arr.shape,dtype="bool")
    if set == 'tr1': 
        rr, cc = draw.disk((514, 458), 401) #((y,x),rad)
        mask[rr, cc] = False
    elif set == 'tr2': 
        #rr, cc = draw.polygon((560,328,766),(25,117,178))
        rr, cc = draw.ellipse(455,175,80,170,rotation=np.deg2rad(45)) #better than polygon
        mask[rr, cc] = False
        rr1, cc1 = draw.disk((512, 488), 405)
        mask[rr1,cc1] =False
    elif set == 'ch1': 
        rr, cc = draw.disk((520, 505), 450)
        mask[rr, cc] = False
    elif set == 'ch2': 
        rr, cc = draw.disk((515, 515), 451)
        mask[rr, cc] = False
    else:
        print('Specify set (tr1, tr2, ch1, or ch2).')
    image_arr_copy[mask] = 1
    return image_arr_copy
    #return mask

def canny_bin(image_arr):
    edge_bin = canny(image_arr,sigma=2)
    edge_bin = util.invert(edge_bin)
    return edge_bin

def water_shed(masked_img):
    cstruct = disk(3)
    s = [[1,1,1],
         [1,1,1],
         [1,1,1]]
    dilated = binary_dilation(masked_img,disk(4))
    distance = ndi.distance_transform_edt(dilated) #Exact euclidean distance transform
    coords = peak_local_max(distance, footprint=cstruct, labels=dilated) #Find peaks in an image
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, c_count = ndi.label(mask,structure=s)
    labeled = watershed(-distance, markers, mask=dilated, compactness=0)
    return labeled, c_count

def expon (x,a,k,c):
        form = a*np.exp(-k*x)+c
        return form

def cell_count(path='', ser='',plot_all=''):
    img_df = load_imgs(path)
    img_df['Binary']=''
    img_df['Masked']=''
    img_df['CellCount']=''
    for n in range(0,len(img_df),1):
        print(f'Processing {img_df.iloc[n,0]} {round(((n+1)/len(img_df))*100,1)}%')
        binary_img = canny_bin(img_df.iloc[n,1])
        img_df.iloc[n,2] = [binary_img]
        masked = mask_set(binary_img,set=ser) #Binary image column 
        masked = util.invert(masked)
        img_df.iloc[n,3] = [masked]  
        labeled, c_count = water_shed(masked)
        img_df.iloc[n,4] = c_count
        #if plot_all=='yes':
        #    fig,ax = plt.subplots(figsize=(12,12))
        #    ax.imshow(img_df.iloc[n,1],cmap='gray')
        #    ax.imshow(labeled,alpha=0.2,cmap='gist_rainbow')
        #    ax.set_title(f'File: {img_df.iloc[n,0]}, Cell count: {c_count}')
        #    #imsave(f'Segmented_{img_df.iloc[n,0]}',fig)
        #    plt.savefig(f'Segmented_{img_df.iloc[n,0]}')
    img_df.to_csv(f'{path}{ser}.csv', index=False)
    
    x = [re.sub(".*/", "", n[-4:]) for n in img_df['FileName']]
    x = np.array(x)
    x = x.astype(int)
    y = img_df['CellCount']
    y = np.array(y)
    y = y.astype(int)
    popt, pcov = curve_fit(expon, x, y, p0=(1, -1e-6, 1))
    k=popt[1] # Get growth constant
    y_x = expon(x,*popt)
    
    fig,ax = plt.subplots(figsize=(12,12))
    ax.plot(x,y)
    ax.plot(x, y_x, 'r-', label='fit, k = %.2f a = %.2f' %(popt[1],popt[0]))
    ax.set_title(f'Series={ser}, k={k}')
    ax.set(xlabel='Image', ylabel='Cell Count')
    return fig

def main(args):
    path = sys.argv[1]
    ser = sys.argv[2]
    main_plot = cell_count(path,ser)
    main_plot.savefig(f'{path}{ser}.png')
    return main_plot
    
if __name__ == "__main__":
    if '-h' in sys.argv:
        usage()
    else:
        main(sys.argv)
