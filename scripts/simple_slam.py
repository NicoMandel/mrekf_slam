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
from mrekf.eval_utils import plot_gt, plot_rs_gt, get_robot_idcs_map, plot_map_est, plot_ellipse, _get_robot_ids, \
get_ATE, get_fp_idcs_map, plot_robs_est, plot_xy_est, _get_xyt_true, get_offset, _get_r_xyt_est, \
get_offset


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
    r2.control = RandomPath(workspace=lm_map, seed=robot.control._seed+1)
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
    # EKF_fp = EKF_FP(x0=x0_inc, P0=P0_inc, sensor=(sensor2, W), robot=(robot, V_r1), history=history,
    #                 fp_list=fp_list, motion_model=mot_model,
    #                 r2=robots)

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
        # EKF_fp=EKF_fp
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
    # dump_json(exp_dict, exp_path)

    ###########################
    # RUN
    ###########################
    f = os.path.join(resultsdir, 'testnew.gif')
    html = ekf.run_animation(T=30, format=None) # format= "gif", file=f)    # format=None
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

    h_mrekf = ekf.history
    h_ekf_i = EKF_include.history
    h_ekf_e = EKF_exclude.history
    # h_ekf_fp = EKF_fp.history

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
    # fp_map_idcs = get_fp_idcs_map(h_ekf_fp, fp_list)
    # plot_map_est(h_ekf_fp, marker=marker_map_est, dynamic_map_idcs=fp_map_idcs, state_length=mot_model.state_length, ellipse=map_est_ell)

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
    # plot_xy_est(h_ekf_fp, **r_est)     
    # plot_ellipse(h_ekf_fp, **covar_r_kws)
    plt.legend()
    plt.show()    

    # Transform from map frame to the world frame -> now changed into three variables
    # calculating ATE from the histories
    x_true = _get_xyt_true(h_mrekf)
    ate_exc_h = get_ATE(h_ekf_e, map_lms=lm_map, x_t=x_true)
    ate_inc_h = get_ATE(h_ekf_i, map_lms=lm_map, x_t=x_true, ignore_idcs=r2_list)
    ekf_ate_h = get_ATE(h_mrekf, map_lms=lm_map, x_t=x_true, ignore_idcs=r2_list)
    # ate_fp_h = get_ATE(h_ekf_fp, map_lms=lm_map, x_t=x_true, ignore_idcs=fp_list)

    print("Mean trajectory error excluding the robot (Baseline): Calculated from histories \t Mean {:.5f}\t std: {:.5f}".format(
        ate_exc_h.mean(), ate_exc_h.std()
    ))
    print("Mean trajectory error including the robot as a static LM (False Negative): Calculated from histories \t Mean {:.5f}\t std: {:.5f}".format(
        ate_inc_h.mean(), ate_inc_h.std()
    ))
    print("Mean trajectory error including the robot as a dynamic LM: Calculated from histories \t Mean {:.5f}\t std: {:.5f}".format(
        ekf_ate_h.mean(), ekf_ate_h.std()
    ))
    # print("Mean trajectory error including a static landmark as dynamic (False Positive): calcualted from histories \t Mean {:.5f}\t std: {:.5f}".format(
    #     ate_fp_h.mean(), ate_fp_h.std()
    # ))


    #calculating absolute difference
    x_est = ekf.get_xyt()
    dist_ekf = get_offset(x_true, x_est)
    
    x_inc = EKF_include.get_xyt()
    x_exc = EKF_exclude.get_xyt()
    dist_inc = get_offset(x_true, x_inc)
    dist_exc = get_offset(x_true, x_exc)

    # x_fp = EKF_fp.get_xyt()
    # dist_fp = get_offset(x_true, x_fp)

    print("Mean real offset excluding the robot (Baseline): \t Mean {:.5f}\t std: {:.5f}".format(
        dist_exc.mean(), dist_exc.std()
    ))
    print("Mean real offset including the robot as a static LM (False Negative): \t Mean {:.5f}\t std: {:.5f}".format(
        dist_inc.mean(), dist_inc.std()
    ))
    print("Mean real offset including the robot as a dynamic LM: \t Mean {:.5f}\t std: {:.5f}".format(
        dist_ekf.mean(), dist_ekf.std()
    ))
    # print("Mean real offset including a static landmark as dynamic (False Positive): \t Mean {:.5f}\t std: {:.5f}".format(
    #     dist_fp.mean(), dist_fp.std()
    # ))
    #calculating absolute difference
    x_est =_get_r_xyt_est(h_mrekf)
    x_inc = _get_r_xyt_est(h_ekf_i)
    x_exc = _get_r_xyt_est(h_ekf_e)
    # x_fp = _get_r_xyt_est(h_ekf_fp)
    dist_ekf_h = get_offset(x_true, np.asarray(x_est))
    dist_inc_h = get_offset(x_true, np.asarray(x_inc))
    dist_exc_h = get_offset(x_true, np.asarray(x_exc))
    # dist_fp_h = get_offset(x_true, np.asarray(x_fp))

    print("Mean real offset excluding the robot (Baseline) - from histories: \t Mean {:.5f}\t std: {:.5f}".format(
        dist_exc_h.mean(), dist_exc_h.std()
    ))
    print("Mean real offset including the robot as a static LM (False Negative)  - from histories: \t Mean {:.5f}\t std: {:.5f}".format(
        dist_inc_h.mean(), dist_inc_h.std()
    ))
    print("Mean real offset including the robot as a dynamic LM  - from histories: \t Mean {:.5f}\t std: {:.5f}".format(
        dist_ekf_h.mean(), dist_ekf_h.std()
    ))
    # print("Mean real offset including a static landmark as dynamic (False Positive)  - from histories: \t Mean {:.5f}\t std: {:.5f}".format(
    #     dist_fp_h.mean(), dist_fp_h.std()
    # ))