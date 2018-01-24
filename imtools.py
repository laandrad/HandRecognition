#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:54:36 2018

@author: alejandro
"""

import numpy as np
import cv2
from keras.models import load_model


class Window():
    def __init__(self, box_height, box_width, frame_height, frame_width, nh, nw):
        
        self.box_height = box_height
        self.box_width = box_width
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.nh = nh
        self.nw = nw
        
    def slide(self, i, j):
        window_height = (self.frame_height - self.box_height) / self.nh
        window_width = (self.frame_width - self.box_width) / self.nw
        height_start = int(i * window_height)
        height_end = int(i * window_height + self.box_height)
        width_start = int(j * window_width)
        width_end = int(j * window_width + self.box_width)
        
        return height_start, height_end, width_start, width_end
    

class Model():
    
    def __init__(self, path, window):
        self.window = window
        
        model = load_model(path)
        self.model = model
    
    
    def predict(self, image, i, j):
        h1, h2, w1, w2 = self.window.slide(i, j)
        img = image[h1:h2, w1:w2]
        prediction = self.model.predict(img.reshape(1,250, 200, 3))
        label = np.argmax(prediction)
        conf = np.max(prediction)
        
        if label == 0 and conf > 0.5:
            cv2.rectangle(image, (w1, h1), (w2, h2), (0,255,0), 1)
        
        return label, conf