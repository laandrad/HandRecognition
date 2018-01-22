#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:08:36 2018

@author: alejandro
"""

import glob
import cv2
import numpy as np

# Get hand images
images = glob.glob('/Users/alejandro/AnacondaProjects/cv/Hands/*.jpg')
images.sort()

img = cv2.imread(images[0], 0)
hand_features = img.flatten()

for i in range(1, len(images)):
    img = cv2.imread(images[i], 0)
    hand_features = np.vstack((hand_features, img.flatten()))

# Save length of hand examples
print(hand_features.shape)
n_hands = hand_features.shape[0]

# Get No_hand images
images = glob.glob('/Users/alejandro/AnacondaProjects/cv/Hands/*.jpg')
images.sort()

img = cv2.imread(images[0], 0)
other_features = img.flatten()

for i in range(1, len(images)):
    img = cv2.imread(images[i], 0)
    other_features = np.vstack((other_features, img.flatten()))

# Save length of other examples
print(other_features.shape)
n_other = other_features.shape[0]

# stack feature matrices
features = np.vstack((hand_features, other_features))

# Create labels
y_hands = np.tile(1, n_hands)
y_other = np.tile(0, n_other)
labels = np.hstack((y_hands, y_other))

print('features shape:', features.shape)
print('labels shape:', labels.shape)

np.savetxt('/Users/alejandro/AnacondaProjects/cv/features.csv',
           features,
           delimiter=',')
np.savetxt('/Users/alejandro/AnacondaProjects/cv/labels.csv',
           labels,
           delimiter=',')