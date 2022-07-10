import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform, FundamentalMatrixTransform
np.set_printoptions(suppress=True)


# turn [[x,y]] => [[x,y,1]]
def add_ones(x):
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

# Identity rotational translation matrix
IRt = np.eye(4)

#pose
def extractRt(F):
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    U,d,Vt = np.linalg.svd(F)
    assert np.linalg.det(U) > 0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U,W.T), Vt)
    t = U[:,2]
    ret = np.eye(4)
    ret[:3,:3] = R
    ret[:3,3] = t
    return ret
    # Rt = np.concatenate([R,t.reshape(3,1)], axis=1)
    # return Rt

# cv2.goodFeaturesToTrack takes grayscale image frame with type np.uint8, so we convert it to grayscale using numpy
# takes max corners, set to 3000 in this case. It also takes a quality level and min distance between corners.
def extract(img):
    orb = cv2.ORB_create()
    
    # detection
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 1000, 0.01, 7)
    
    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1],size=20) for f in pts]
    kps, des = orb.compute(img, kps)
    
    # return pts and des
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des


def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:,0:2]

def denormalize(K, pt):
    ret =  np.dot(K, np.array([pt[0], pt[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))

def match_frames(f1,f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)
    
    # Lowe's ration test
    ret = []
    idx1, idx2 = [], []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]

            # travel less than 10% of diagonal and be within orb distance 32
            if np.linalg.norm((p1-p2)) < 0.2*np.linalg.norm([f1.w, f1.h]) and m.distance < 32:
                # keep around indices
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                ret.append((p1, p2))

    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    #normalize coords
 

    # fit matrix
    model, inliers = ransac((ret[:, 0],ret[:, 1]),
                            FundamentalMatrixTransform,
                            # EssentialMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.001,
                            # residual_threshold=1.0,
                            max_trials=100)
    
    print(f"Matches: {len(f1.des)} -> {len(matches)} -> {len(inliers)} -> {sum(inliers)}")
    # ignore outliers
    Rt = extractRt(model.params)

    # return
    return idx1[inliers], idx2[inliers], Rt

class Frame(object):
    def __init__(self, mapp, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.pose = IRt
        self.h, self.w = img.shape[0:2]
        pts, self.des = extract(img)
        self.pts = normalize(self.Kinv, pts)

        
        self.id = len(mapp.frames)
        mapp.frames.append(self)

    #    ret[:,0,:] = self.normalize(ret[:,0,:])
    # ret[:,1,:] = self.normalize(ret[:,1,:])


    

    

