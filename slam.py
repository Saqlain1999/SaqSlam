#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np
from display import Display
from frame import denormalize, Frame, match_frames
from pointmap import Map, Point


# set this!
F = int(os.getenv("F", "800"))

# Creating a object of display class, it actually tas frames from video and makes it viewable using Sdl2

# Feature Extractor Class returns good features to track using cv2.ORB, it's method extract takes image as a parameter (the same one from display class)
# Camera
W = 1920//2
H = 1080//2
K = np.array(([F,0,W//2],[0,F,H//2],[0,0,1]))
Kinv = np.linalg.inv(K)


# main classes
mapp = Map(K) 
if os.getenv("D3D") is not None:
    mapp.create_viewer()


display = None
if os.getenv("D2D") is not None:
    display = Display(W,H) 


def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = p[0][0]*pose1[2] - pose1[0]
        A[1] = p[0][1]*pose1[2] - pose1[1]
        A[2] = p[1][0]*pose2[2] - pose2[0]
        A[3] = p[1][1]*pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] =  vt[3]
    return ret

def process_frame(img):
    img = cv2.resize(img, (W,H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return
    print(f"\n*** frame {frame.id} ***")
    
    # match with previous frame
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]
    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)


    for i,idx in enumerate(idx2):
        if f2.pts[idx] is not None:
            f2.pts[idx].add_observation(f1, idx1[i])

    # homogenous 3-D coords
    pts4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])
    pts4d /= pts4d[:, 3:]
    
    # reject pts without enough depth
    # reject pts behind the camera
    unmatched_points = np.array([f1.pts[i] is None for i in idx1])
    print(f"Adding: {np.sum(unmatched_points)} points")
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_points


    for i,p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])



    for pt1,pt2 in zip(f1.kps[idx1], f2.kps[idx2]):
        u1,v1 = denormalize(K, pt1)
        u2,v2 = denormalize(K, pt2)
        cv2.circle(img, (u1, v1), color=(0,255,0),radius=3)
        cv2.line(img, (u1, v1), (u2,v2), color=(255,0,0))
    
    # 2-D
    if display is not None:
        display.show(img)
    
    # Optimize the mapp
    if frame.id >= 4:
        mapp.optimize()
    
    # 3-D
    mapp.display()
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"{sys.argv[0]} <video.mp4>")
        exit(-1)

    cap = cv2.VideoCapture(sys.argv[1])
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        else:
            break
    