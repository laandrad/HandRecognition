#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:20:33 2018

@author: alejandro
"""

import os
import itertools
import progressbar
import cv2
import numpy as np
import pandas as pd

def preprocess(paths, labels, path_out):
    for i in range(len(paths)):
        copy_to_path(paths[i], labels[i], path_out)


def copy_to_path(path, label, path_out): 
    
    train, dev, test = tdt_split(path)
    os.makedirs(path_out + 'train/' + label, exist_ok=True)
    os.makedirs(path_out + 'dev/' + label, exist_ok=True)
    os.makedirs(path_out + 'test/' + label, exist_ok=True)
        
    for file in train:
        copy_reduce_image(path, train, path_out + 'train/' + label + '/')
    for file in dev:
        copy_reduce_image(dev, path_out + 'dev/' + label + '/')
    for file in test:
        copy_reduce_image(test, path_out + 'test/' + label + '/')
 
    
def copy_reduce_image(path_in, folder, path_out):
    print("{} images in folder {}".format(folder.size, path_in))
    print("copying to folder", path_out)
    bar = progressbar.ProgressBar(max_value=folder.size)
    files = list(folder)
    for i in range(folder.size):
#        print(path_out + files[i])
#            img = cv2.imread(path_in + files[i])
#            img = cv2.resize(img, (100, 125))
#            cv2.imwrite(path_out + files[i], img)
        bar.update(i)


def tdt_split(path):
    files = os.listdir(path)
    files = [x for x in files if x.endswith('.jpg')]
    files = pd.Series(files)
    subsample = pd.Series(np.random.choice(3, len(files), p=[0.7, 0.15, 0.15]))
    df = pd.DataFrame({'files': files, 'sample': list(subsample)})
    train = df.files[df['sample'] == 0]
    dev = df.files[df['sample'] == 1]
    test = df.files[df['sample'] == 2]
    
    return train, dev, test
 
path1 = '/Users/alejandro/Dropbox (Work)/Hand_Data/Hands/'
path2 = '/Users/alejandro/Dropbox (Work)/Hand_Data/No_hands/'
path_out = '/Users/alejandro/Dropbox (Work)/Hand_Data/'
paths = [path1, path2]
labels = ['hand', 'no_hand']

copy_to_path(paths[0], labels[0], path_out)

