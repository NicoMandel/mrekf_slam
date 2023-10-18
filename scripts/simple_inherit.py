"""
    basic example using inherited functions.
    Todo in order of priority
    1. check the normal simulation run works - no errors
    2. adapt sensor model to work with dictionary
    3. check histories can be used to create evaluations 
    4. include logging module for debugging purposes
"""
import os.path
import numpy as np
from datetime import date, datetime
import matplotlib.pyplot as plt

from roboticstoolbox import LandmarkMap, Bicycle, RandomPath
from math import pi

# own import
from mrekf.simulation import Simulation
from mrekf.ekf_base import BasicEKF
from mrekf.dynamic_ekf import Dynamic_EKF
from mrekf.sensor import  RobotSensor, get_sensor_model
from mrekf.motionmodels import StaticModel, KinematicModel, BodyFrame
from mrekf.utils import convert_experiment_to_dict, dump_pickle, dump_json
from mrekf.eval_utils import plot_gt, plot_rs_gt, get_robot_idcs_map, plot_map_est, plot_ellipse, _get_robot_ids, \
get_fp_idcs_map, plot_robs_est, plot_xy_est

if __name__=="__main__":
    # general experimental setting
    history=True
    verbose=True                # todo use this to set the loglevel
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
	# Setup secondary Robots    
    # additional_marker= VehicleMarker()
    sec_robots = {}
    robot_offset = 100
    for i in range(1):
        V_r2 = np.diag([0.2, np.deg2rad(5)]) ** 2
        r2 = Bicycle(covar=V_r2, x0=(1, 4, np.deg2rad(45)), animation="car")
        r2.control = RandomPath(workspace=lm_map, seed=robot.control._seed+1)
        r2.init()
        sec_robots[i + robot_offset] = r2
    # robots = [r2]
    rg = 10

    # Setup state estimate - is only robot 1!
    x0_est =  np.array([0., 0., 0.])      # initial estimate
    P0 = np.diag([0.05, 0.05, np.deg2rad(0.5)]) ** 2

    # Estimate the second robot
    V_est = np.diag([0.3, 0.3]) ** 2
    V_est_kin = np.zeros((4,4))
    V_est_kin[2:, 2:] = V_est
    # mot_model = StaticModel(V_est)
    # mot_model = KinematicModel(V=V_est_kin, dt=robot.dt)
    V_est_bf = V_est_kin.copy()
    mot_model = BodyFrame(V_est_bf, dt=robot.dt)
    sensor2 = get_sensor_model(mot_model, robot=robot, r2=sec_robots, covar= W, lm_map=lm_map, rng = rg, angle=[-pi/2, pi/2])

    ##########################
    # EKF SETUPS
    ##########################
    # excluding -> base ekf
    x0_exc = x0_est.copy()
    P0_exc = P0.copy()
    ekf_exc = BasicEKF(x0_exc, P0_exc, robot=(robot, V_r1), sensor=(sensor2, W),
                       ignore_ids=list(sec_robots.keys()),
                       history=history)

    # including -> base ekf
    x0_inc = x0_est.copy()    
    P0_inc = P0.copy()
    ekf_inc = BasicEKF(x0_exc, P0_exc, robot=(robot, V_r1), sensor=(sensor2, W),
                       ignore_ids=[],
                       history=history)
    
    # Dynamic EKFs
    # FP -> dynamic Ekf    
    x0_fp = x0_est.copy()
    P0_fp = P0.copy()
    fp_list = [2]
    ekf_fp = Dynamic_EKF(
        x0=x0_fp, P0=P0_fp, robot=(robot, V_r1), sensor = (sensor2, W),
        motion_model=mot_model, dynamic_ids=fp_list, ignore_ids=list(sec_robots.keys()),
        history=history
    )

    # real one
    ekf_mr = Dynamic_EKF(
        x0=x0_est, P0=P0, robot=(robot, V_r1), sensor=(sensor2, W),
        motion_model=mot_model, dynamic_ids=list(sec_robots.keys()),
        history=history
    )
    
    ekf_list = [
        ekf_exc,
        ekf_inc,
        ekf_fp,
        ekf_mr
    ]
    
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
        "robots" : sec_robots,
        "sensor" : sensor2,
        "motion_model" : mot_model,
        "map" : lm_map,
        "FP" : fp_list 
    }
    fpath = os.path.join('results', 'sometest', 'blabla.json')
    # dump_json(sdict, fpath)

    # write the experiment settings first
    # exp_dict = convert_experiment_to_dict(sdict)
    exp_path = os.path.join(resultsdir, dname + ".json")
    # dump_json(exp_dict, exp_path)

    ###########################
    # RUN
    ###########################
    sim = Simulation(
        robot=(robot, V_r1),
        r2=sec_robots,
        P0=P0,      # not used, only for inheritance
        sensor=(sensor2, W),
        verbose=verbose,
        history=history,
        ekfs=ekf_list
        )
    f = os.path.join(resultsdir, 'newtest.mp4')
    html = sim.run_animation(T=30, format=None) #format="mp4", file=f) # format=None
    plt.show()
    # HTML(html)

    #####################
    ## SECTION ON SAVING
    ######################
    # Write the files

    # dump_pickle(ekf.history, resultsdir, name="MREKF")
    # dump_pickle(EKF_include.history, resultsdir, name="EKF_inc")
    # dump_pickle(EKF_exclude.history, resultsdir,  name="EKF_exc")
    # dump_pickle(EKF_fp.history, resultsdir,  name="EKF_fp")

    h_mrekf = ekf_mr.history
    h_ekf_i = ekf_inc.history
    h_ekf_e = ekf_exc.history
    h_ekf_fp = ekf_fp.history

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
    plot_gt(hist=h_mrekf, **r_dict)
    r_dict["color"] = "b"
    r_dict["label"] = "r2 true"
    plot_rs_gt(h_mrekf, **r_dict)
    marker_map_est = map_markers
    marker_map_est["color"] = "b"
    marker_map_est["label"] = "map est mr"
    marker_map_est["marker"] = "x"
    map_est_ell = {
        "color" : "b",
        "linestyle" : ":"
    }
    map_idcs_dyn = get_robot_idcs_map(h_mrekf)
    plot_map_est(h_mrekf, dynamic_map_idcs = map_idcs_dyn, state_length=mot_model.state_length, marker=marker_map_est, ellipse=map_est_ell)
    marker_map_est["color"] = "y"
    marker_map_est["label"] = "map est inc"
    map_est_ell["color"] = "y"
    plot_map_est(h_ekf_i, marker=marker_map_est, ellipse = map_est_ell)
    marker_map_est["color"] = map_est_ell["color"] = "g"
    marker_map_est["label"] = "map est exc"
    plot_map_est(h_ekf_e, marker=marker_map_est)
    marker_map_est["color"] = map_est_ell["color"] = "m"
    marker_map_est["label"] = "map est fp"
    fp_map_idcs = get_fp_idcs_map(h_ekf_fp, fp_list)
    plot_map_est(h_ekf_fp, marker=marker_map_est, dynamic_map_idcs=fp_map_idcs, state_length=mot_model.state_length, ellipse=map_est_ell)

    # Plotting path estimates
    r_est = {
        "color" : "r",
        "linestyle" : "-.",
        "label" : "r est"
    }
    covar_r_kws ={
        "color" : "r",
        "linestyle" : ":",
    }
    plot_xy_est(h_mrekf, **r_est)
    plot_ellipse(h_mrekf, **covar_r_kws)
    r2_est = {
        "color" : "b",
        "linestyle" : "dotted",
        "marker" : ".",
        "label" : "r2 est"
    }
    covar_r2_kws = {
                "color" : "b",
                "linestyle" : ":",
            }
    plot_robs_est(h_mrekf, **r2_est)
    r2_list = _get_robot_ids(h_mrekf) 
    for r in r2_list:
        covar_r2_kws["label"] = "r{} covar".format(r)
        plot_ellipse(h_mrekf, r, **covar_r2_kws)
    # excluding
    r_est["color"] = covar_r_kws["color"] = "g"
    r_est["label"] = "r est exc"
    plot_xy_est(h_ekf_e, **r_est)
    plot_ellipse(h_ekf_e, **covar_r_kws)
    # including
    r_est["color"] = covar_r_kws["color"] = "y"
    r_est["label"] = "r est inc"
    plot_xy_est(h_ekf_i, **r_est)
    plot_ellipse(h_ekf_i, **covar_r_kws)
    # FPs
    r_est["color"] = covar_r_kws["color"] = "m"
    r_est["label"] = "r est fp"
    plot_xy_est(h_ekf_fp, **r_est)     
    plot_ellipse(h_ekf_fp, **covar_r_kws)
    plt.legend()
    plt.show()    