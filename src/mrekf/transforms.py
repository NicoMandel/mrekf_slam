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