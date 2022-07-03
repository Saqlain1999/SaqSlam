#!/usr/bin/env python

import time
import cv2
import numpy as np
from display import Display
from featureExtractor import FeatureExtractor

# Video:
videoFromDir = './video2.mp4'

# Screen Size
W = 1920//2
H = 1080//2

F = 1
# Creating a object of display class, it actually tas frames from video and makes it viewable using Sdl2
display = Display(W,H)

K = np.array(([F,0,W//2],[0,F,H//2],[0,0,1]))

# Feature Extractor Class returns good features to track using cv2.ORB, it's method extract takes image as a parameter (the same one from display class)
fe = FeatureExtractor(K)



def process_frame(img):
    img = cv2.resize(img, (W,H))
    matches = fe.extract(img)
    
    print("%d matches" % (len(matches)))

    for pt1, pt2 in matches:
            u1,v1 = fe.denormalize(pt1)
            u2,v2 = fe.denormalize(pt2)


            cv2.circle(img, (u1, v1), color=(0,255,0),radius=3)
            cv2.line(img, (u1, v1), (u2,v2), color=(255,0,0))
    
    display.show(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture(videoFromDir)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        else:
            break
    