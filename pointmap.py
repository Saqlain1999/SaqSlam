from multiprocessing import Process, Queue
import numpy as np
import pypangolin as pango
import OpenGL.GL as gl 
import g2o
from frame import poseRt
class Map(object):
    def __init__(self, K):
        self.frames = []
        self.points = []
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.state = None
        self.q = None

    # Optimizer g2opy
    def optimize(self):

        # create g2o optimizer
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)
        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
        

        # add frames to graph
        for f in self.frames:
            pose = f.pose
            # pose = np.linalg.inv(pose)
            sbacam = g2o.SBACam(g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3]))
            # sbacam.set_cam(1.0,1.0,0.0,0.0,1.0)
            sbacam.set_cam(f.K[0][0],f.K[1][1],f.K[0][2],f.K[1][2],1.0)
            v_se3 = g2o.VertexCam()
            v_se3.set_id(f.id)
            v_se3.set_estimate(sbacam)
            v_se3.set_fixed(f.id <= 1)
            opt.add_vertex(v_se3)


        # add points to frames
        PT_ID_OFFSET = 0x10000
        for p in self.points:
            pt = g2o.VertexSBAPointXYZ()
            pt.set_id(p.id + PT_ID_OFFSET)
            pt.set_estimate(p.pt[0:3])
            pt.set_marginalized(True)
            pt.set_fixed(False)
            opt.add_vertex(pt)

            for f in p.frames:
                edge = g2o.EdgeProjectP2MC()
                edge.set_vertex(0, pt)
                edge.set_vertex(1, opt.vertex(f.id))
                edge.set_measurement(f.kpus[f.pts.index(p)])
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)

        # Optimize                
        opt.set_verbose(True)
        opt.initialize_optimization()
        # init g2o optimizer
        opt.optimize(50)  

        # put frames back
        for f in self.frames:
            est = opt.vertex(f.id).estimate()
            R = est.rotation().matrix()
            t = est.translation()
            f.pose = poseRt(R, t)

        # put edges back
        for p in self.points:
            est = opt.vertex(p.id + PT_ID_OFFSET).estimate()
            p.pt = np.array(est)

            # p.pt = est


    # Viewer
    def create_viewer(self):
        self.q = Queue()
        self.vp = Process(target=self.viewer_thread, args=(self.q,))
        self.vp.daemon = True
        self.vp.start()

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
            pango.ModelViewLookAt(0, -10, -8,
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
        if self.state is None or not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0,0.0,0.0,1.0)
        self.dcam.Activate(self.scam)

        # draw poses
        gl.glColor3f(0.0, 1.0, 0.0)
        for pose in self.state[0]:
            pango.glDrawFrustum(self.Kinv, 1000, 1000, pose, 1)

        # draw keypoints
        gl.glPointSize(5)
        gl.glColor3f(1.0,1.0,1.0)
        pango.glDrawPoints(self.state[1])
        pango.glDrawPoints(self.state[2])
            
        pango.FinishFrame()
        
 

    def display(self):
        if self.q is None:
            return
        poses, pts, colors = [], [], []
        for f in self.frames:
            poses.append(f.pose)
            # poses.append(np.linalg.inv(f.pose))
        for p in self.points:
            pts.append(p.pt)
            colors.append(p.color)
        self.q.put((poses, np.array(pts), np.array(colors)/256.0))


class Point(object):
    # A point is a 3-D point in the world
    # Each point is observed in multiple Frames

    def __init__(self, mapp, loc, color):
        self.pt = loc
        self.frames = []
        self.idxs = []
        self.color = np.copy(color)

        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)