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
        
        
    def create_grid(self):
        window_height = (self.frame_height - self.box_height) / self.nh
        window_width = (self.frame_width - self.box_width) / self.nw
        
        h1, h2, w1, w2 = [], [], [], []
        for i in range(self.nh+1):
            for j in range(self.nw+1):
                h1.append(int(i * window_height))
                h2.append(int(i * window_height + self.box_height))
                w1.append(int(j * window_width))
                w2.append(int(j * window_width + self.box_width))

        return h1, h2, w1, w2
        
        
    def slide(self, image, h1, h2, w1, w2):
        
        batch = np.zeros((1, 250, 200, 3))
        
        for i in range(self.nh+1):
            for j in range(self.nw+1):
                img = image[h1[i]:h2[i], w1[j]:w2[j]].reshape(1,250, 200, 3)
                batch = np.concatenate((batch, img), axis=0)
        
        return np.delete(batch, 0, 0)
    

class Model():
    
    def __init__(self, path, window):
        self.window = window
        
        model = load_model(path)
        self.model = model
    
    
    def predict(self, batch):
        predictions = self.model.predict(batch)
        return predictions
                
    
    def bounding_box(self, predictions, image, h1, h2, w1, w2):
        
        for i in range(len(predictions)):
            if np.argmax(predictions[i]) == 0:
                cv2.rectangle(image, 
                              (w1[i], h1[i]), 
                              (w2[i], h2[i]), 
                              (0,255,0), 
                              1)