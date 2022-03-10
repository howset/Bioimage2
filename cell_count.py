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
    print('Usage: python3 cell_count.py Req1 Req2')
    print('Req1: path to images')
    print('Req2: image set (tr1, tr2, ch1, or ch2)')
    #print('Opt3: plot individual image (non functional)')
    print('')

def load_imgs(path):
    '''
    Loads (.tif) images in a directory and puts them in a dataframe.

    Parameters
    ----------
    path : Directory where the image files are contained

    Returns
    -------
    df_img : A dataframe with filename and the image array.
    '''
    
    files_img = glob.glob(f'{path}*.tif')
    files_img.sort()
    df_img = pd.DataFrame(columns=['FileName','Image'])
    for file in files_img:
        print(f'Loading {file}') 
        filename = re.sub(".*/", "", file[:-4])
        
        # tifffile plugin for loading compressed tiffs requires imagecodecs
        img = imread(file,plugin='tifffile') 
        df_img.loc[len(df_img)] = [filename,img]
    return df_img

def mask_set(image_arr,set=''):
    '''
    Masking function for a binarized image, specific to certain set(s).

    Parameters
    ----------
    image_arr : The binarized image array.
    set : Specific set ('tr1', 'tr2', 'ch1', or 'ch2').

    Returns
    -------
    image_arr_copy : The resulting masked image.

    '''
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
        #ellipse is better than polygon
        rr, cc = draw.ellipse(455,175,80,170,rotation=np.deg2rad(45)) 
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
    '''
    Edge detection to binarize the image.

    Parameters
    ----------
    image_arr : The image array.

    Returns
    -------
    edge_bin : The resulting binarized image.

    '''
    edge_bin = canny(image_arr,sigma=2)
    edge_bin = util.invert(edge_bin)
    return edge_bin

def water_shed(masked_img):
    '''
    Watershedding and labeling.

    Parameters
    ----------
    masked_img : Self explanatory.

    Returns
    -------
    labeled : Watershedded & labeled image (segmented).
    c_count : Number of segments.

    '''
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
    '''
    Exponential growth function.

    Parameters
    ----------
    x : x-axis
    a : constant
    k : growth constant
    c : constant

    Returns
    -------
    form : Self explanatory.

    '''
    form = a*np.exp(k*x)+c
    return form

def cell_count(path='', ser='',plot_all=''):
    img_df = load_imgs(path)
    img_df['Binary']=''
    img_df['Masked']=''
    img_df['CellCount']=''
    img_df['Area']=''
    for n in range(0,len(img_df),1):
        print(f'Processing {img_df.iloc[n,0]} {round(((n+1)/len(img_df))*100,1)}%')
        binary_img = canny_bin(img_df.iloc[n,1])
        img_df.iloc[n,2] = [binary_img]
        masked = mask_set(binary_img,set=ser) #Binary image column 
        masked = util.invert(masked)
        img_df.iloc[n,3] = [masked]  
        labeled, c_count = water_shed(masked)
        img_df.iloc[n,4] = c_count
        img_df.iloc[n,5] = np.sum(masked)
        #if plot_all=='yes':
        #    fig,ax = plt.subplots(figsize=(12,12))
        #    ax.imshow(img_df.iloc[n,1],cmap='gray')
        #    ax.imshow(labeled,alpha=0.2,cmap='viridis')
        #    ax.set_title(f'File: {img_df.iloc[n,0]}, Cell count: {c_count}')
        #    #imsave(f'Segmented_{img_df.iloc[n,0]}',fig)
        #    plt.savefig(f'Segmented_{img_df.iloc[n,0]}')
    img_df.to_csv(f'{path}{ser}.csv', index=False)
    
    x = [re.sub('.*/', '', n[-4:]) for n in img_df['FileName']]
    x = np.array(x)
    x = x.astype(int)
    y = img_df['CellCount']
    y = np.array(y)
    y = y.astype(int)
    popt, pcov = curve_fit(expon, x, y, p0=(1, 1e-6, 1))
    k = popt[1] # Get growth constant
    a = popt[0]
    y_x = expon(x,*popt)
    
    y_A = img_df['Area']
    y_A = np.array(y_A)
    y_A = y_A.astype(int)
    popt_A, pcov_A = curve_fit(expon, x, y_A, p0=(1, 1e-6, 1))
    k_A = popt_A[1] # Get growth constant
    a_A = popt_A[0]
    yA_x = expon(x,*popt_A)
    
    fig,ax = plt.subplots(1,2,figsize=(12,8))
    ax = ax.ravel()
    ax[0].plot(x,y)
    ax[0].plot(x, y_x, 'r-', label=f'fit, k = {round(k,4)} a = {round(a,4)}')
    ax[0].legend(loc="upper left")
    ax[0].set_title(f'Count. k={round(k,4)}')
    ax[0].set(xlabel='Image', ylabel='Cell Count')
    ax[1].plot(x,y_A)
    ax[1].plot(x, yA_x, 'r-', label=f'fit, k = {round(k_A,4)} a = {round(a_A,4)}')
    ax[1].legend(loc="upper left")
    ax[1].set_title(f'Area. k={round(k_A,4)}')
    ax[1].set(xlabel='Image', ylabel='Area')
    fig.suptitle(f'Series={ser}')
    return fig

def main(args):
    '''
    Call the cell_count function and save the resulting plot.

    Parameters
    ----------
    args : Path and specific set.

    Returns
    -------
    main_plot : The exponential growth plot and fit.

    '''
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
