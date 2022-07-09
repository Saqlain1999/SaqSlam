#!/usr/bin/env python

import os
import time
import cv2
import numpy as np
from display import Display
from frame import denormalize, Frame, match_frames, IRt
import g2o
from pointmap import Map, Point


np.set_printoptions(suppress=True)


# Video:
videoFromDir = './video2.mp4'

# Creating a object of display class, it actually tas frames from video and makes it viewable using Sdl2

# Feature Extractor Class returns good features to track using cv2.ORB, it's method extract takes image as a parameter (the same one from display class)
# Camera
W = 1920//2
H = 1080//2
F = 270
K = np.array(([F,0,W//2],[0,F,H//2],[0,0,1]))
Kinv = np.linalg.inv(K)


# main classes
mapp = Map(Kinv)
display = Display(W,H) if os.getenv("D2D") is not None else None

def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

def process_frame(img):
    img = cv2.resize(img, (W,H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]
    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)
    
    # homogenous 3-D coords
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3:]
    
    # reject pts without enough depth
    # reject pts behind the camera
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)


    for i,p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])



    for pt1,pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1,v1 = denormalize(K, pt1)
        u2,v2 = denormalize(K, pt2)
        cv2.circle(img, (u1, v1), color=(0,255,0),radius=3)
        cv2.line(img, (u1, v1), (u2,v2), color=(255,0,0))
    
    # 2-D
    if display is not None:
        display.show(img)
    
    # 3-D
    mapp.display()
    

if __name__ == "__main__":
    cap = cv2.VideoCapture(videoFromDir)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        else:
            break
    