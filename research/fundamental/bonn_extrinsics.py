"""Bonn dataset motion-capture marker -> RGB-D camera correction.
Selected by training split and tested against heldout depth in bonn_calibration_audit.py.
Do not reuse for datasets with different hardware.
"""
import numpy as np
M=np.array([[1.0157,.1828,-.2389,.0113],
            [.0009,-.8431,-.6413,-.0098],
            [-.3009,.6147,-.8085,.0111],
            [0,0,0,1.]],dtype=float)
R_ROS=np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1.]],dtype=float)
u,s,v=np.linalg.svd(M[:3,:3])
M[:3,:3]=u@v
EXTRINSIC=R_ROS@M
def camera_pose_from_marker(T):
    return np.asarray(T)@EXTRINSIC
