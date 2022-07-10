import pypangolin as pango
import OpenGL.GL as gl 
from multiprocessing import Process, Queue
import numpy as np


class Map(object):
    def __init__(self, Kinv):
        self.frames = []
        self.points = []
        self.Kinv = Kinv
        self.state = None
        self.q = Queue()
        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()

    # Running Threads
    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pango.CreateWindowAndBind("ZlykhSLAM", w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        self.scam = pango.OpenGlRenderState(
            pango.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
            pango.ModelViewLookAt(0, -5, -8,
                                  0, 0, 0,
                                  0, -1, 0))

        self.handler = pango.Handler3D(self.scam)

        self.dcam = (
            pango.CreateDisplay()
            .SetBounds(
            pango.Attach(0),
            pango.Attach(1),
            pango.Attach(0),
            pango.Attach(1),
            -w / h,
        )
            .SetHandler(self.handler))
    def viewer_refresh(self, q):
        if self.state is None or q.empty():
            self.state = q.get()

        # turn state into points
        # ppts = np.array([d[:3, 3] for d in self.state[0]])
        spts = np.array([d[:3] for d in self.state[1]])


        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)


        gl.glColor3f(0.0, 1.0, 0.0)
        for pose in self.state[0]:
            pango.glDrawFrustum(self.Kinv, 0, 0, pose, 1)

        # pango.glDrawPoints([d[:3,3] for d in self.state[0]])

        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pango.glDrawPoints(spts)
        
        pango.FinishFrame()
        


    def display(self):
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        self.q.put((poses, pts))


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