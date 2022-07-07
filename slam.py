#!/usr/bin/env python

import time
import cv2
import numpy as np
from display import Display
from frame import denormalize, Frame, match_frames, IRt
import g2o
from multiprocessing import Process, Queue
import pypangolin as pango
import OpenGL.GL as gl 
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

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []

        # create viewer process
        self.q = Queue()
        self.viewer = Process(target=self.viewer_thread, args=(self.q,))
        self.viewer.Daemon = True
        self.viewer.start()


    def viewer_thread(self, q):
        pango.CreateWindowAndBind("ZlykhSLAM", 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        self.scam = pango.OpenGlRenderState(
            pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pango.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pango.AxisY))

        self.handler = pango.Handler3D(self.scam)

        self.dcam = (
            pango.CreateDisplay()
            .SetBounds(
            pango.Attach(0),
            pango.Attach(1),
            pango.Attach.Pix(0),
            pango.Attach(1),
            -640.0 / 480.0,
        )
            .SetHandler(self.handler))
        darr = None
        if darr is None or not q.empty():
            darr = q.get(True)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)
        gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)
        pango.glDrawPoints([d[:3, 1] for d in darr[0]])

        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pango.glDrawPoints([d[:3] for d in darr[1]])
        pango.FinishFrame()
    # def viewer_refresh(self, q):
        # for d in darr[1]:
            # pango.glDrawPoints(d, dtype=np.float64)
        # pango.glDrawPoints(dtype=np.float64, data=(d for d in darr[1]))

        # pango glDrawPoints
        


    def display(self):
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        self.q.put((poses, pts))
        self.viewer_thread(self.q)

# main classes
display = Display(W,H)
mapp = Map()


    # for p in points:
    #     print(p.xyz)

class Point(object):
    # A point is a 3-D point in the world
    # Each point is observed in multiple Frames

    def __init__(self, mapp, loc):
        self.pt = loc
        self.frames = []
        self.idxs = []
        
        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)

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
    