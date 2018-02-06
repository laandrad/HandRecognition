#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:25:53 2018

@author: alejandro
"""
import time
import cv2
from imtools import pyramid, sliding_window
from sklearn.externals import joblib
from keras.models import load_model


# Initialize video capture, window, and CNN model
cap = cv2.VideoCapture(0)

box_height = 90
box_width = 90
frame_height = 450
frame_width = 630
step_size = 90

model_cnn = load_model('model_cnn.h5')
model_svm = joblib.load('svm_model.pkl')


# Start video feed
while True:
    # read a frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (frame_width, frame_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tic = time.time()
    for (i, resized) in enumerate(pyramid(gray)):
        for (x, y, w) in sliding_window(resized, step_size, box_width, box_height):
            if w.shape[1] != box_width or w.shape[0] != box_height:
                continue
            # img = cv2.resize(w, (90, 90))
            img = img.reshape(1, 90, 90, 1)//255
            cnn_features = model_cnn.predict(img)
            pred = model_svm.predict(cnn_features)
            # print('prediction:', pred[0])
            cv2.rectangle(frame, (x, y),
                          (x + box_width, y + box_height),
                          (0, 255, 0),
                          1)

            if pred[0] == 0:
                print('there\'s a hand!')
                # cv2.imshow('hand', w)
                cv2.rectangle(frame, (x, y),
                              (x + box_height, y + box_width),
                              (255, 255, 255),
                              3)
    toc = time.time()
    grid_time = toc - tic
    print('Execution time:', grid_time)
    
    cv2.imshow('Window 1', frame)
        
    time.sleep(.16)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
exit()
