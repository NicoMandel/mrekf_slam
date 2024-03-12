import numpy as np
from spatialmath import SE2

def forward(x : np.ndarray, q : np.ndarray) -> np.ndarray:
    """
        forward transformation of q by x.
        q is 2 x N
        x is a (3,) np array in form of (x,y,theta)  
    """
    x_se = SE2(x)
    x_n = x_se * q
    return x_n.squeeze()

def inverse(x : np.ndarray, q : np.ndarray) -> np.ndarray:
    """
        inverse transformation of q by x.
        q is 2 x N
        x is a (3,) np array in form of (x,y,theta) 
    """
    x_se = SE2(x)
    x_n = x_se.inv() * q
    return x_n.squeeze()

def tf_from_tR(t_e : np.ndarray, R_e : np.ndarray) -> np.ndarray:
    """
        returns a [x, y, theta] transform to be used with forward and inverse
    """
    ang = _get_angle(R_e)
    tf = np.r_[
        t_e,
        ang
    ]
    return tf

def _get_angle(R : np.ndarray) -> float:
    """
        Function to get an angle from a rotation matrix.
        based on:
        spatialmath/pose2d.py L. 203
        [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/spatialmath/pose2d.py#L.203]]
    """
    s = R[0,1]
    c = R[0,0]
    ang = np.arctan2(s, c)
    ang = np.rad2deg(ang)
    return ang

def _get_rotation_offset(rot : np.ndarray) -> float:
    """
        Function to get the absolute rotation of a map transform
    """
    if isinstance(rot, np.ndarray):
        ang = _get_angle(rot)
    else:
        ang = rot
    return np.abs(ang)

def _get_translation_offset(t : np.ndarray) -> float:
    """
        Function to get the absolute translation offset of a map transform
    """
    dist = np.linalg.norm(t)
    return dist

def get_transform_offsets(tf : np.ndarray) -> tuple[float, float]:
    """
        Wrapper function to return rotation and translation offsets. uses sub functions above.
        tf is of type np.ndarray with format [x, y, theta]
        returns in order
    """

    r_dist = _get_rotation_offset(tf[2])
    t_dist = _get_translation_offset(tf[:2])
    return t_dist, r_dist