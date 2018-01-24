#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:25:53 2018

@author: alejandro
"""
import itertools
import time
import cv2
from imtools import Window, Model


# Initialize video capture, window, and CNN model
cap = cv2.VideoCapture(0)

box_height = 250
box_width = 200
frame_height = 360
frame_width = 640
nh = 2
nw = 4
window = Window(box_height, box_width, frame_height, frame_width, nh, nw)

model = Model('model.hd5', window)


# Start video feed
while(True):
    # read a frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 360))
    
    tic = time.time()
    [model.predict(frame, i, j) for i in range(nh+1) for j in range(nw+1)]
    toc = time.time()
    print("Execution time: {}".format(toc - tic))
    
    cv2.imshow('frame', frame)
        
    time.sleep(10)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
exit()
