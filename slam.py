#!/usr/bin/env python

import time
import cv2
import numpy as np
from display import Display
from frame import denormalize, Frame, match
import g2o

# Video:
videoFromDir = './video2.mp4'

# Creating a object of display class, it actually tas frames from video and makes it viewable using Sdl2

# Feature Extractor Class returns good features to track using cv2.ORB, it's method extract takes image as a parameter (the same one from display class)
# Camera
W = 1920//2
H = 1080//2
F = 270
K = np.array(([F,0,W//2],[0,F,H//2],[0,0,1]))

# main classes
display = Display(W,H)
# fe = FeatureExtractor()

frames = []
def process_frame(img):
    img = cv2.resize(img, (W,H))
    frame = Frame(img, K)
    frames.append(frame)
    if len(frames) <= 1:
        return
    ret, Rt = match(frames[-1], frames[-2])

    for pt1,pt2 in ret:
        u1,v1 = denormalize(K, pt1)
        u2,v2 = denormalize(K, pt2)
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
    