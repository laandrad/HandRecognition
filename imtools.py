#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:54:36 2018

@author: alejandro
"""
import cv2


def pyramid(image, scale=1.5, min_size=(90, 90)):
    yield image

    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (h, w))

        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        yield image


def sliding_window(image, step_size, box_width, box_height):
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield (x, y, image[y:y + box_height, x:x + box_width])
