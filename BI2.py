#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 13:53:58 2022

@author: howset
"""

##############################
## Import packages ###########
##############################

import glob
import pandas as pd
from skimage.io import imread,imshow

##############################
## Functions #################
##############################

def load_imgs(path):
    files_img = glob.glob(path)
    files_img.sort()
    df_img = pd.DataFrame(columns=['Images'])
    for file in files_img:
        print(file) # to show progress
        img = imread(file) # tifffile plugin for loading compressed tiffs requires imagecodecs
        df_img.iloc[len(df_img)] = [img]
        #imshow(img)
    return(df_img)


##############################
## Procedure #################
##############################

path = ('/home/howset/workspace/Bioimage2/Images/training_BF-C2DL-HSC/01/*.tif') # E14 path
imgs = load_imgs(path)

imgs.loc[1]
imshow(imgs.loc[1])
