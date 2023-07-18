"""
    basic example from Peter
"""

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import RVC3 as rvc
from IPython.display import HTML

from roboticstoolbox import LandmarkMap, Bicycle, RandomPath, RangeBearingSensor
from math import pi

# own import
from utils import EKF_MR, RobotSensor, EKF_base


if __name__=="__main__":
    # Setup robot 1
    V_r1 = np.diag([0.02, np.deg2rad(0.5)]) ** 2
    V_r2 = np.diag([0.02, np.deg2rad(0.5)]) ** 2
    robot = Bicycle(covar=V_r1, x0=(0, 0, np.deg2rad(0.1)), 
            animation="car")
    # setup map - used for workspace config
    map = LandmarkMap(20, workspace=10)
    robot.control = RandomPath(workspace=map)
    # Setup Sensor
    W = np.diag([0.1, np.deg2rad(1)]) ** 2
    # sensor = RangeBearingSensor(robot=robot, map=map, covar=W,		# ! map is a property of sensor here. not of EKF 
            # range=4, angle=[-pi/2, pi/2])
	# Setup Robot 2
    r2 = Bicycle(covar=V_r2, x0=(1, 4, np.deg2rad(45)))
    r2.control = RandomPath(workspace=map,seed=robot.control._seed+1)
    robots = [r2]
    sensor = RobotSensor(robot=robot, r2 = robots, map=map, covar = W, range=10, angle=[-pi/2, pi/2])

    # Setup state estimate - is only robot 1!
    x0_est =  np.array([0., 0., 0.])      # initial estimate
    P0 = np.diag([0.05, 0.05, np.deg2rad(0.5)]) ** 2
    # estimate of the robots movement
    # TODO: make sure these are set right
    V_est = np.diag([0.3, 0.3]) ** 2

    # include 2 other EKFs of type EKF_base
    history=True
    x0_inc = x0_est.copy()
    x0_exc = x0_est.copy()
    P0_inc = P0.copy()
    P0_exc = P0.copy()
    # EKFs also include the robot and the sensor - but not to generate readings or step, only to get the associated V and W
    # and make use of h(), f(), g(), y() and its derivatives
    EKF_include = EKF_base(x0=x0_inc, P0=P0_inc, sensor=(sensor, W), robot=(robot, V_r1), history=history)  # EKF that includes the robot as a static landmark 
    EKF_exclude = EKF_base(x0=x0_exc, P0=P0_exc, sensor=(sensor, W), robot=(robot, V_r1), history=history)  # EKF that excludes the robot as a landmark

    ekf = EKF_MR(
        robot=(robot, V_r1),
        r2=robots,
        P0=P0,
        sensor=(sensor, W),
        V2=V_est,
        verbose=True,
        history=True,
        # extra parameters
        EKF_include = EKF_include,
        EKF_exclude = EKF_exclude      
        )

    # Run
    html = ekf.run_animation(T=5,format=None) #format=None)
    plt.show()
    # HTML(html)

    # Plotting
    map.plot();       # plot true map
    # plt.show()
    robot.plot_xy();  # plot true path
    r2.plot_xy()
    ekf.plot_map();      # plot estimated landmark position
    ekf.plot_ellipse();  # plot estimated covariance
    ekf.plot_xy();       # plot estimated robot path
    plt.show()

    # Transform from map frame to the world frame
    T = ekf.get_transform(map)

