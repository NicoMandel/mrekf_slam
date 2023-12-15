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
np.set_printoptions(precision=4, suppress=True, linewidth=10000, edgeitems=30)
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
from mrekf.utils import convert_simulation_to_dict, dump_json, dump_ekf, dump_gt

from mrekf.eval_utils import plot_xy_est, plot_map_est, plot_dyn_gt, plot_gt, plot_ellipse, get_dyn_lms, get_dyn_idcs_map, plot_dyn_est

if __name__=="__main__":
    # general experimental setting
    history=True
    verbose=True                # todo use this to set the loglevel
    store = False
    seed = 1
    np.random.seed(seed)
    robot_offset = 100
    rg = 50

    s_lms = 2
    d_lms = 1

    # Setup robot 1
    V_r1 = np.diag([0.2, np.deg2rad(5)]) ** 2
    robot = Bicycle(covar=V_r1, x0=(0, 0, np.deg2rad(0.1)), 
            animation="car")
    
    # setup map - used for workspace config
    lm_map = LandmarkMap(s_lms, workspace=10)
    robot.control = RandomPath(workspace=lm_map, seed=seed)
    
	# Setup secondary Robots    
    sec_robots = {}
    for i in range(d_lms):
        V_r2 = np.diag([0.2, np.deg2rad(5)]) ** 2
        r2 = Bicycle(covar=V_r2, x0=(np.random.randint(-10, 10), np.random.randint(-10, 10), np.deg2rad(np.random.randint(0,360))), animation="car")
        r2.control = RandomPath(workspace=lm_map, seed=None)
        r2.init()
        sec_robots[i + robot_offset] = r2
    
    # Setup estimate functions for the second robot. the sensor depends on this!
    V_est_bf = V_r2.copy()             # best case - where the model is just like the real thing
    V_est_stat = np.diag([0.2, 0.2]) ** 2
    V_est_kin = V_est_stat.copy()
    # mot_model = StaticModel(V_est_stat)
    # mot_model = KinematicModel(V=V_est_kin, dt=robot.dt)
    mot_model = BodyFrame(V_est_bf, dt=robot.dt)
    
    # Setup Sensor
    W = np.diag([0.4, np.deg2rad(10)]) ** 2
    sensor2 = get_sensor_model(mot_model, robot=robot, lm_map=lm_map,
                               r2=sec_robots, covar= W, 
                               rg = rg, angle=[-pi, pi])

    ##########################
    # EKF SETUPS
    ##########################
    # Setup state estimate - is only robot 1!
    x0_est =  np.array([0., 0., 0.])      # initial estimate
    P0 = np.diag([0.05, 0.05, np.deg2rad(0.5)]) ** 2
    
    # excluding -> basic ekf
    x0_exc = x0_est.copy()
    P0_exc = P0.copy()
    ekf_exc = BasicEKF(
        description="EKF_EXC",
        x0=x0_exc, P0=P0_exc, robot=(robot, V_r1), sensor=(sensor2, W),
        ignore_ids=list(sec_robots.keys()),
        history=history
    )

    # including -> basic ekf
    x0_inc = x0_est.copy()    
    P0_inc = P0.copy()
    ekf_inc = BasicEKF(
        description="EKF_INC",
        x0=x0_inc, P0=P0_inc, robot=(robot, V_r1), sensor=(sensor2, W),
        ignore_ids=[],
        history=history
    )
    
    # Dynamic EKFs
    # FP -> dynamic Ekf    
    x0_fp = x0_est.copy()
    P0_fp = P0.copy()
    fp_list = [1]
    ekf_fp = Dynamic_EKF(
        description="EKF_FP",
        x0=x0_fp, P0=P0_fp, robot=(robot, V_r1), sensor = (sensor2, W),
        motion_model=mot_model,
        dynamic_ids=fp_list,
        ignore_ids=list(sec_robots.keys()),
        history=history
    )

    # real one
    ekf_mr = Dynamic_EKF(
        description="EKF_MR",
        x0=x0_est, P0=P0, robot=(robot, V_r1), sensor=(sensor2, W),
        motion_model=mot_model,
        dynamic_ids=list(sec_robots.keys()),
        history=history
    )
    
    ekf_list = [
        ekf_exc,
        ekf_inc,
        ekf_fp,
        ekf_mr
    ]

    ###########################
    # RUN
    ###########################
    bdir = os.path.dirname(__file__)
    pdir = os.path.abspath(os.path.join(bdir, '..'))
    rdir = os.path.join(pdir, 'results', "20231106_42_33")
    simfpath = os.path.join(rdir, 'config.json')

    sim = Simulation(
        robot=(robot, V_r1),
        r2=sec_robots,
        P0=P0,      # not used, only for inheritance
        sensor=(sensor2, W),
        verbose=verbose,
        history=history,
        ekfs=ekf_list
    )
    simdict = convert_simulation_to_dict(sim, seed=seed)
    
    videofpath = os.path.join(rdir, 'debug.mp4')
    # html = sim.run_animation(T=30, format=None) # format=None format="mp4", file=videofpath
    sim.run_simulation(T=30)
    # plt.show()
    # HTML(html)

    #####################
    ## SECTION ON SAVING
    ######################
    
    ############################# 
    # SAVE Experiment
    #############################
    # get a dictionary out from the simulation to store
    if store:
        dump_json(simdict, simfpath)
        dump_gt(sim.history, rdir)
        for ekf in ekf_list:
            dump_ekf(ekf.history, ekf.description, rdir)
            # else:
            #     hlms = ekf.history[-1].landmarks
            #     print("Test debug line")
    # else plot directly - to check if something goes wrong with saving!
    else:
        plt.figure(figsize=(16,10))

        # Splitting the histories and settings
        h_ekfmr = ekf_mr.history
        h_ekfinc = ekf_inc.history
        h_ekfexc = ekf_exc.history
        h_ekffp = ekf_fp.history
        gt_hist = sim.history

        cfg_ekfmr = simdict["EKF_MR"]
        cfg_ekfinc = simdict["EKF_INC"]
        cfg_ekfexc = simdict["EKF_EXC"]
        cfg_ekffp = simdict["EKF_FP"]

        # lm_vis(h_ekfmr)
        # check_angle(h_ekfmr, 100)
        
        ##############################
        # PLOTTING Experiment
        ##############################
        workspace = np.array(simdict['map']['workspace'])
        mp = np.array(simdict['map']['landmarks'])
        lm_map = LandmarkMap(map=mp, workspace=workspace)
        # Plotting Ground Truth
        map_markers = {
            "label" : "map true",
            "marker" : "+",
            "markersize" : 10,
            "color" : "black",
            "linewidth" : 0
        }
        lm_map.plot(**map_markers);       # plot true map
        r_dict = {
            "color" : "black",
            "label" : "r true"
            }
        plot_gt(hist=gt_hist, **r_dict)
        r_dict["color"] = "b"
        r_dict["label"] = "r2 true"
        plot_dyn_gt(hist_gt=gt_hist, **r_dict)

        # Plotting the Map estimates
        marker_map_est = map_markers
        marker_map_est["color"] = "r"
        marker_map_est["label"] = "map est mr"
        marker_map_est["marker"] = "x"
        map_est_ell = {
            "color" : "r",
            "linestyle" : ":"
        }
        
        map_idcs_dyn = get_dyn_idcs_map(cfg_ekfmr , h_ekfmr)
        ekf_mr_mmsl = cfg_ekfmr["motion_model"]["state_length"]   
        plot_map_est(h_ekfmr, dynamic_map_idcs = map_idcs_dyn, state_length=ekf_mr_mmsl, marker=marker_map_est, ellipse=map_est_ell)
        # plot_map_est(h_ekfmr, marker=marker_map_est, ellipse = map_est_ell)
        marker_map_est["color"] = "y"
        marker_map_est["label"] = "map est inc"
        map_est_ell["color"] = "y"
        plot_map_est(h_ekfinc, marker=marker_map_est, ellipse = map_est_ell)
        marker_map_est["color"] = map_est_ell["color"] = "g"
        marker_map_est["label"] = "map est exc"
        plot_map_est(h_ekfexc, marker=marker_map_est, ellipse = map_est_ell)
        marker_map_est["color"] = map_est_ell["color"] = "m"
        marker_map_est["label"] = "map est fp"
        ekf_fp_mmsl = cfg_ekffp["motion_model"]["state_length"]
        fp_map_idcs = get_dyn_idcs_map(cfg_ekffp, h_ekffp)
        plot_map_est(h_ekffp, marker=marker_map_est, dynamic_map_idcs=fp_map_idcs, state_length=ekf_fp_mmsl, ellipse=map_est_ell)
        # plot_map_est(h_ekffp, marker=marker_map_est, ellipse = map_est_ell)

        # Plotting path estimates
        r_est = {
            "color" : "r",
            "linestyle" : "-.",
            "label" : "r est mr"
        }
        covar_r_kws ={
            "color" : "r",
            "linestyle" : ":",
        }
        plot_xy_est(h_ekfmr, **r_est)
        plot_ellipse(h_ekfmr, **covar_r_kws)

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
        plot_dyn_est(h_ekfmr, cfg_ekfmr, **r2_est)
        r2_list = get_dyn_lms(cfg_ekfmr) 
        for r in r2_list:
            covar_r2_kws["label"] = "r:{} covar".format(r)
            plot_ellipse(h_ekfmr, r, **covar_r2_kws)
        
        # excluding
        r_est["color"] = covar_r_kws["color"] = "g"
        r_est["label"] = "r est exc"
        plot_xy_est(h_ekfexc, **r_est)
        plot_ellipse(h_ekfexc, **covar_r_kws)

        # including
        r_est["color"] = covar_r_kws["color"] = "y"
        r_est["label"] = "r est inc"
        plot_xy_est(h_ekfinc, **r_est)
        plot_ellipse(h_ekfinc, **covar_r_kws)
        
        # FPs
        r_est["color"] = covar_r_kws["color"] = "m"
        r_est["label"] = "r est fp"
        plot_xy_est(h_ekffp, **r_est)     
        plot_ellipse(h_ekffp, **covar_r_kws)
        plt.legend()
        loc = os.path.join(rdir, 'debug_full.jpg')
        # plt.savefig(loc, dpi=400)
        plt.show()    

    print("Test Debug line")
    
