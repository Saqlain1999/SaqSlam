# ZlykhSLAM

A basic implementation of SLAM using python

Libraries Used
-----

* sdl2 for 2-D display
* cv2 for feature tracking and matching
* g2opy for optimization
* skimage for ransac and EssentialMatrixTransform
* pypangolin for 3-D display

To Run
-----

* Requires a video to run SLAM on, this implementaiton is just for learning this is a python based SLAM so it isn't efficient
* Just provide it the video with runing slam.py file
  
Usage
-----
```
export D2D=1        # 2-D viewer
export D3D=1        # 3-D viewer
export REVERSE=1    # Reverse Video
export F=270        # Focal Length (in pixels)

./slam.py <video.mp4>
```
### Example

```
REVERSE=1 F=270 D3D=1 D2D=1 ./slam.py videos/test_countryroad_reversed.mp4
```

TODO
-----

* Add Optimizer for F
* Replace essential matrix for pose estimation once we have a track
 * Add kinematic model
 * Run g2o to only optimize the latest pose
* Add search by projection o re-find old map points
 * Check if points are in the field of view of the camera
* Improve init to not need REVERSE environment variable
* Add multiscale feature extractor
