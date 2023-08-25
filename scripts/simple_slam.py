"""
    basic example from Peter
    ! The best evaluation is whether K @ innovation is smaller in the first 3 states -> indicator of improvement of the estimate
"""
import os.path
import numpy as np
from datetime import date, datetime
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import RVC3 as rvc
from IPython.display import HTML

from roboticstoolbox import LandmarkMap, Bicycle, RandomPath, RangeBearingSensor, VehicleMarker
from math import pi

# own import
from mrekf.mr_ekf import EKF_MR
from mrekf.sensor import  RobotSensor, get_sensor_model
from mrekf.ekf_base import  EKF_base
from mrekf.motionmodels import StaticModel, KinematicModel, BodyFrame
from mrekf.ekf_fp import EKF_FP
from mrekf.utils import convert_experiment_to_dict, dump_pickle, dump_json


if __name__=="__main__":
    seed = 0
    # Setup robot 1
    V_r1 = np.diag([0.2, np.deg2rad(5)]) ** 2
    robot = Bicycle(covar=V_r1, x0=(0, 0, np.deg2rad(0.1)), 
            animation="car")
    # setup map - used for workspace config
    lm_map = LandmarkMap(20, workspace=10)
    robot.control = RandomPath(workspace=lm_map, seed=seed)
    # Setup Sensor
    W = np.diag([0.2, np.deg2rad(3)]) ** 2
    # sensor = RangeBearingSensor(robot=robot, map=map, covar=W
            # range=4, angle=[-pi/2, pi/2])
	# Setup Robot 2
    # additional_marker= VehicleMarker()
    V_r2 = np.diag([0.2, np.deg2rad(5)]) ** 2
    r2 = Bicycle(covar=V_r2, x0=(1, 4, np.deg2rad(45)), animation="car")
    r2.control = RandomPath(workspace=lm_map,seed=robot.control._seed+1)
    r2.init()
    robots = [r2]

    rg = 10
    sensor = RobotSensor(robot=robot, r2 = robots, lm_map=lm_map, covar = W, range=rg, angle=[-pi/2, pi/2])

    # Setup state estimate - is only robot 1!
    x0_est =  np.array([0., 0., 0.])      # initial estimate
    P0 = np.diag([0.05, 0.05, np.deg2rad(0.5)]) ** 2

    # Estimate the second robot
    V_est = np.diag([0.3, 0.3]) ** 2
    # mot_model = StaticModel(V_est)

    V_est_kin = np.zeros((4,4))
    V_est_kin[2:, 2:] = V_est
    # mot_model = KinematicModel(V=V_est_kin, dt=robot.dt)
    V_est_bf = V_est_kin.copy()
    mot_model = BodyFrame(V_est_bf, dt=robot.dt)
    sensor2 = get_sensor_model(mot_model, robot=robot, r2=robots, covar= W, lm_map=lm_map, rng = rg, angle=[-pi/2, pi/2])

    # include 2 other EKFs of type EKF_base
    history=True
    x0_inc = x0_est.copy()
    x0_exc = x0_est.copy()
    P0_inc = P0.copy()
    P0_exc = P0.copy()
    x0_fp = x0_est.copy()
    P0_fp = P0.copy()
    # EKFs also include the robot and the sensor - but not to generate readings or step, only to get the associated V and W
    # and make use of h(), f(), g(), y() and its derivatives
    EKF_include = EKF_base(x0=x0_inc, P0=P0_inc, sensor=(sensor2, W), robot=(robot, V_r1), history=history)  # EKF that includes the robot as a static landmark
    EKF_exclude = EKF_base(x0=x0_exc, P0=P0_exc, sensor=(sensor2, W), robot=(robot, V_r1), history=history)  # EKF that excludes the robot as a landmark
    fp_list = [2]
    EKF_fp = EKF_FP(x0=x0_inc, P0=P0_inc, sensor=(sensor2, W), robot=(robot, V_r1), history=history,
                    fp_list=fp_list, motion_model=mot_model,
                    r2=robots)

    ekf = EKF_MR(
        robot=(robot, V_r1),
        r2=robots,
        P0=P0,
        sensor=(sensor2, W),
        motion_model=mot_model,
        verbose=False,          # True
        history=True,
        # extra parameters
        EKF_include = EKF_include,
        EKF_exclude = EKF_exclude,
        EKF_fp=EKF_fp
        )
    
    #############################
    #   saving experiment
    #############################
    bdir = os.path.dirname(__file__)
    pdir = os.path.abspath(os.path.join(bdir, '..'))
    rdir = os.path.join(pdir, 'results')
    # dname = "{}-{}-{}-{}".format(len(robots), type(mot_model).__name__,len(lm_map), 10)
    dname = "testres"
    resultsdir = os.path.join(rdir, dname)
    sdict = {
        "robot" : robot,
        "seed" : seed,
        "robots" : robots,
        "sensor" : sensor2,
        "motion_model" : mot_model,
        "map" : lm_map,
        "FP" : fp_list 
    }
    fpath = os.path.join('results', '1-BodyFrame-20-10', 'blabla.json')
    # dump_json(sdict, fpath)

    # write the experiment settings first
    exp_dict = convert_experiment_to_dict(sdict)
    exp_path = os.path.join(resultsdir, dname + ".json")
    dump_json(exp_dict, exp_path)

    ###########################
    # RUN
    ###########################
    f = os.path.join(resultsdir, 'test.gif')
    html = ekf.run_animation(T=30,format= "gif", file=f) #format=None)
    plt.show()
    # HTML(html)

    #####################
    ## SECTION ON SAVING
    ######################
    # Write the files

    dump_pickle(ekf.history, resultsdir, name="MREKF")
    dump_pickle(EKF_include.history, resultsdir, name="EKF_inc")
    dump_pickle(EKF_exclude.history, resultsdir,  name="EKF_exc")
    dump_pickle(EKF_fp.history, resultsdir,  name="EKF_fp")

    #####################
    # SECTION ON PLOTTING
    #####################

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
        "linestyle" : "dotted",
        "marker" : ".",
        "label" : "r2 est"
    }
    ekf.plot_robot_xy(r_id=0+100, **r2_est) # todo - check the todo in this function - just plot the robot when it has been observed at least once - change logging for this
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
    marker_fp = {
            "marker": "x",
            "markersize": 10,
            "color": "m",
            "linewidth": 0,
            "label" : "map est fp"
    }
    EKF_include.plot_map(marker=marker_inc)
    EKF_exclude.plot_map(marker=marker_exc)
    EKF_fp.plot_map(marker=marker_fp)
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
    fp_r = {
        "color" : "m",
        "label" : "r est fp",
        "linestyle" : "-."
    }
    EKF_exclude.plot_xy(**exc_r)
    EKF_include.plot_xy(**inc_r)
    EKF_fp.plot_xy(**fp_r)
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
    covar_fp_kws = {
        "color" : "m",
        "linestyle" : ":",
        "label" : "fp covar"
    }
    EKF_exclude.plot_ellipse(**covar_exc_kws)
    EKF_include.plot_ellipse(**covar_inc_kws)
    EKF_fp.plot_ellipse(**covar_fp_kws)
    covar_fp_kws = {
        "color" : "m",
        "linestyle" : ":",
        "label" : "lm {} fp covar".format(fp_list[0])
    }
    EKF_fp.plot_robot_estimates(**covar_fp_kws)

    
    plt.legend()
    plt.show()

    # displaying covariance
    ekf.disp_P()
    plt.show()

    # Evaluation section
    # Testing the Pnorms
    Pnorm_hist = ekf.get_Pnorm()
    lm_id_late = 7       # 7 only seen after a while
    r_id = 0 + 100
    t = 25
    # print(ekf.get_Pnorm_lm(0))
    # print(ekf.get_Pnorm_lm(0, t))
    # print(ekf.get_Pnorm_lm(lm_id_late))
    # print(ekf.get_Pnorm_lm(lm_id_late, t))

    ekf.get_Pnorm_r(r_id)
    ekf.get_Pnorm_r(r_id, t)

    # inspecting the estimated robots variables over time:
    r_index = ekf.robot_index(list(ekf.seen_robots.keys())[0])
    state_len = mot_model.state_length
    r_list = np.array([h.xest[r_index : r_index + state_len] for h in ekf.history if len(h.xest) > r_index])
    plt.figure()
    plt.plot(r_list[:,0], label="x")
    plt.plot(r_list[:,1], label="y")
    plt.plot(r_list[:,2], label="v")
    plt.plot(r_list[:,3], label="theta")
    plt.legend()
    plt.show()
    
    # Transform from map frame to the world frame -> now changed into three variables
    # calculating ate
    ate_exc = EKF_exclude.get_ATE(map_lms=lm_map)
    ate_inc = EKF_include.get_ATE(map_lms=lm_map)
    ekf_ate = ekf.get_ATE(map_lms=lm_map)
    ate_fp = EKF_fp.get_ATE(map_lms=lm_map)

    print("Mean trajectory error excluding the robot (Baseline): \t Mean {:.5f}\t std: {:.5f}".format(
        ate_exc.mean(), ate_exc.std()
    ))
    print("Mean trajectory error including the robot as a static LM (False Negative): \t Mean {:.5f}\t std: {:.5f}".format(
        ate_inc.mean(), ate_inc.std()
    ))
    print("Mean trajectory error including the robot as a dynamic LM: \t Mean {:.5f}\t std: {:.5f}".format(
        ekf_ate.mean(), ekf_ate.std()
    ))
    print("Mean trajectory error including a static landmark as dynamic (False Positive): \t Mean {:.5f}\t std: {:.5f}".format(
        ate_fp.mean(), ate_fp.std()
    ))


    #calculating absolute difference
    x_true = robot.x_hist
    x_est = ekf.get_xyt()
    dist_ekf = EKF_base.get_offset(x_true, x_est)
    
    x_inc = EKF_include.get_xyt()
    x_exc = EKF_exclude.get_xyt()
    dist_inc = EKF_base.get_offset(x_true, x_inc)
    dist_exc = EKF_base.get_offset(x_true, x_exc)

    x_fp = EKF_fp.get_xyt()
    dist_fp = EKF_base.get_offset(x_true, x_fp)

    print("Mean real offset excluding the robot (Baseline): \t Mean {:.5f}\t std: {:.5f}".format(
        dist_exc.mean(), dist_exc.std()
    ))
    print("Mean real offset including the robot as a static LM (False Negative): \t Mean {:.5f}\t std: {:.5f}".format(
        dist_inc.mean(), dist_inc.std()
    ))
    print("Mean real offset including the robot as a dynamic LM: \t Mean {:.5f}\t std: {:.5f}".format(
        dist_ekf.mean(), dist_ekf.std()
    ))
    print("Mean real offset including a static landmark as dynamic (False Positive): \t Mean {:.5f}\t std: {:.5f}".format(
        dist_fp.mean(), dist_fp.std()
    ))