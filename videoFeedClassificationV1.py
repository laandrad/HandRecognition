#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:25:53 2018

@author: alejandro
"""
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
h1, h2, w1, w2 = window.create_grid()

model = Model('model.hd5', window)


# Start video feed
while(True):
    # read a frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (frame_width, frame_height))
    
    tic = time.time()
    batch = window.slide(frame, h1, h2, w1, w2)
    toc = time.time()
    grid_time = toc - tic
    tic = time.time()
    predictions = model.predict(batch)
    toc = time.time()
    prediction_time = toc - tic
    tic = time.time()
    predictions = [(0, 0) for x in range(len(h1))]
    model.bounding_box(predictions, frame, h1, h2, w1, w2)
    toc = time.time()
    drawing_time = toc - tic
    tot_time = grid_time + prediction_time + drawing_time
    
    print("Total Execution time: {},\n Grid: {}, {}%,\n Prediction: {}, {}%,\n Drawing: {}, {}%".format(
            round(tot_time, 2),
            round(grid_time, 2), round(grid_time / tot_time * 100),
            round(prediction_time, 2), round(prediction_time / tot_time * 100),
            round(drawing_time, 2), round(drawing_time / tot_time * 100)))
    
    cv2.imshow('Window 1', frame)
        
    time.sleep(10)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
exit()
