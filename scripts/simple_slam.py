"""
    basic example from Peter
"""

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import RVC3 as rvc
from IPython.display import HTML

from roboticstoolbox import LandmarkMap, Bicycle, RandomPath, RangeBearingSensor, VehicleMarker
from math import pi

# own import
from mrekf.utils import EKF_MR, RobotSensor, EKF_base


if __name__=="__main__":
    # Setup robot 1
    V_r1 = np.diag([0.02, np.deg2rad(0.5)]) ** 2
    V_r2 = np.diag([0.02, np.deg2rad(0.5)]) ** 2
    robot = Bicycle(covar=V_r1, x0=(0, 0, np.deg2rad(0.1)), 
            animation="car")
    # setup map - used for workspace config
    lm_map = LandmarkMap(20, workspace=10)
    robot.control = RandomPath(workspace=lm_map)
    # Setup Sensor
    W = np.diag([0.1, np.deg2rad(1)]) ** 2
    # sensor = RangeBearingSensor(robot=robot, map=map, covar=W,		# ! map is a property of sensor here. not of EKF 
            # range=4, angle=[-pi/2, pi/2])
	# Setup Robot 2
    # additional_marker= VehicleMarker()
    r2 = Bicycle(covar=V_r2, x0=(1, 4, np.deg2rad(45)), animation="car")
    r2.control = RandomPath(workspace=lm_map,seed=robot.control._seed+1)
    r2.init()
    robots = [r2]

    rg = 10
    sensor = RobotSensor(robot=robot, r2 = robots, map=lm_map, covar = W, range=rg, angle=[-pi/2, pi/2])

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
    W2 = W.copy()
    rgb_sens = RangeBearingSensor(robot=robot, map=lm_map, covar=W2, range=rg, angle=[-pi/2, pi/2])
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
    html = ekf.run_animation(T=10,format=None) #format=None)
    plt.show()
    # HTML(html)

    # Plotting Ground Truth
    map_markers = {
        "label" : "map true",
        "marker" : "+",
        "markersize" : 10,
        "color" : "black",
        "linewidth" : 0
    }
    lm_map.plot(**map_markers);       # plot true map
    # plt.show()
    r_dict = {
        "color" : "r",
        "label" : "r true"
        }
    robot.plot_xy(**r_dict);  # plot true path

    r2_dict = {
        "color" : "b",
        "label" : "r2 true"
    }
    r2.plot_xy(**r2_dict)

    marker_map_est = {
            "marker": "x",
            "markersize": 10,
            "color": "b",
            "linewidth": 0,
            "label" : "map est"
    }
    ekf.plot_map(marker=marker_map_est);      # plot estimated landmark position
    # Plotting estimates
    r_est = {
        "color" : "r",
        "linestyle" : "-.",
        "label" : "r est"
    }
    ekf.plot_xy(**r_est);       # plot estimated robot path
    r2_est = {
        "color" : "b",
        "linestyle" : "-.",
        "label" : "r2 est"
    }
    ekf.plot_robot_xy(r_id=0, **r2_est) # todo - check the todo in this function - just plot the robot when it has been observed at least once - change logging for this
    # ekf.plot_robot_estimates(N=20)
    
    # Plotting things
    marker_inc = {
                "marker": "x",
                "markersize": 10,
                "color": "y",
                "linewidth": 0,
                "label" : "map est inc"
            }
    marker_exc = {
            "marker": "x",
            "markersize": 10,
            "color": "g",
            "linewidth": 0,
            "label" : "map est exc"
    }
    EKF_include.plot_map(marker=marker_inc)
    EKF_exclude.plot_map(marker=marker_exc)
    exc_r = {
        "color" : "g",
        "label" : "r est exc",
        "linestyle" : "-."
    }
    inc_r = {
        "color" : "y",
        "label" : "r est inc",
        "linestyle" : "-."
    }
    EKF_exclude.plot_xy(**exc_r)
    EKF_include.plot_xy(**inc_r)

    ## Plotting covariances
    covar_r_kws ={
        "color" : "r",
        "linestyle" : ":",
        "label" : "r covar"
    }
    covar_r2_kws = {
        "color" : "b",
        "linestyle" : ":",
        "label" : "r2 covar"
    }
    ekf.plot_ellipse(**covar_r_kws);  # plot estimated covariance
    ekf.plot_robot_estimates(**covar_r2_kws)

    # baselines
    covar_exc_kws = {
        "color" : "g",
        "linestyle" : ":",
        "label" : "exc covar"
    }
    covar_inc_kws = {
        "color" : "y",
        "linestyle" : ":",
        "label" : "inc covar"

    }
    EKF_exclude.plot_ellipse(**covar_exc_kws)
    EKF_include.plot_ellipse(**covar_inc_kws)

    
    plt.legend()
    plt.show()

    # displaying covariance
    ekf.disp_P()
    plt.show()

    # Evaluation section
    # Testing the Pnorms
    Pnorm_hist = ekf.get_Pnorm()
    lm_id_late = 7       # 7 only seen after a while
    r_id = 0
    t = 25
    print(ekf.get_Pnorm_lm(0))
    print(ekf.get_Pnorm_lm(0, t))
    print(ekf.get_Pnorm_lm(lm_id_late))
    print(ekf.get_Pnorm_lm(lm_id_late, t))

    ekf.get_Pnorm_r(r_id)
    ekf.get_Pnorm_r(r_id, t)
    
    # Transform from map frame to the world frame -> now changed into three variables
    # calculating ate
    t_test = slice(50)
    ate_exc = EKF_exclude.get_ATE(map_lms=lm_map)
    ate_inc = EKF_include.get_ATE(map_lms=lm_map)
    ekf_ate = ekf.get_ATE(map_lms=lm_map) 

    print("Mean trajectory error excluding the robot (Baseline): \t Mean {:.5f}\t std: {:.5f}".format(
        ate_exc.mean(), ate_exc.std()
    ))
    print("Mean trajectory error including the robot as a static LM (False Negative): \t Mean {:.5f}\t std: {:.5f}".format(
        ate_inc.mean(), ate_inc.std()
    ))
    print("Mean trajectory error including the robot as a dynamic LM: \t Mean {:.5f}\t std: {:.5f}".format(
        ekf_ate.mean(), ekf_ate.std()
    ))


    #calculating absolute difference
    x_true = robot.x_hist
    x_est = ekf.get_xyt()
    dist_ekf = EKF_base.get_offset(x_true, x_est)
    
    x_inc = EKF_include.get_xyt()
    x_exc = EKF_exclude.get_xyt()
    dist_inc = EKF_base.get_offset(x_true, x_inc)
    dist_exc = EKF_base.get_offset(x_true, x_exc)

    print("Mean real offset excluding the robot as a static LM (Baseline): \t Mean {:.5f}\t std: {:.5f}".format(
        dist_exc.mean(), dist_exc.std()
    ))
    print("Mean real offset including the robot as a static LM (False Negative): \t Mean {:.5f}\t std: {:.5f}".format(
        dist_inc.mean(), dist_inc.std()
    ))
    print("Mean real offset including the robot as a dynamic LM: \t Mean {:.5f}\t std: {:.5f}".format(
        dist_ekf.mean(), dist_ekf.std()
    ))

    
    

    

