"""
    basic example from Peter
"""

import numpy as np
import matplotlib.pyplot as plt

import RVC3 as rvc
from IPython.display import HTML

from roboticstoolbox import LandmarkMap, Bicycle, RandomPath, RangeBearingSensor
from math import pi

# own import
from utils import EKF_MR


if __name__=="__main__":
    # Setup
    V = np.diag([0.02, np.deg2rad(0.5)]) ** 2
    map = LandmarkMap(20, workspace=10)
    # map.plot()
    W = np.diag([0.1, np.deg2rad(1)]) ** 2 
    robot = Bicycle(covar=V, x0=(3, 6, np.deg2rad(-45)), 
            animation="car")
    robot.control = RandomPath(workspace=map)
    W = np.diag([0.1, np.deg2rad(1)]) ** 2
    sensor = RangeBearingSensor(robot=robot, map=map, covar=W, 
            range=4, angle=[-pi/2, pi/2])
    P0 = np.diag([0.05, 0.05, np.deg2rad(0.5)]) ** 2
    ekf = EKF_MR(robot=(robot, V), P0=P0, sensor=(sensor, W))

    # Run
    html = ekf.run_animation(T=20, format=None)
    # plt.show()
    # HTML(html)

    # Plotting
    map.plot();       # plot true map
    # plt.show()
    # robot.plot_xy();  # plot true path
    ekf.plot_map();      # plot estimated landmark position
    ekf.plot_ellipse();  # plot estimated covariance
    ekf.plot_xy();       # plot estimated robot path
    plt.show()

    # Transform from map frame to the world frame
    T = ekf.get_transform(map)

