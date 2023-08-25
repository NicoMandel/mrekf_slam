import os.path
import numpy as np
from mrekf.utils import load_json, load_pickle
from roboticstoolbox import LandmarkMap
from mrekf.eval_utils import plot_gt, plot_rs_gt, plot_map_est, get_robot_idcs_map, get_fp_idcs_map, \
plot_xy_est, plot_robs_est, plot_ellipse, _get_robot_ids, disp_P, \
get_ATE, _get_xyt_true, get_offset, _get_robot_xyt_est
import matplotlib.pyplot as plt

"""
    Todo: refactor
    FP and MREKF should be treated equally.
        Should only receive a list of indices which are to be treated as dynamic landmarks
        already written in plot_map_est
        _get_state_idx() as function that gets the index of an id in the state vector
    -> fuse self.robots and self.landmarks - because the robot ids are > 100, this should work and it should be a single lookup!
    * calculate differences in K -> for certain phases!    
    """

if __name__=="__main__":
    fig = plt.figure(figsize=(30,14))
    fpath = os.path.dirname(__file__)
    bpath = os.path.abspath(os.path.join(fpath, '..'))
    
    rpath = os.path.join(bpath, "results")
    rdir = "testres"
    exp_path = os.path.join(rpath, rdir, rdir + ".json")
    expd = load_json(exp_path)

    hpath_mrekf = os.path.join(rpath, rdir, "MREKF.pkl")
    hpath_ekf_fp = os.path.join(rpath, rdir, "EKF_exc.pkl")
    hpath_ekf_e = os.path.join(rpath, rdir, "EKF_fp.pkl")
    hpath_ekf_i = os.path.join(rpath, rdir, "EKF_inc.pkl")
    h_mrekf = load_pickle(hpath_mrekf, mrekf = True)
    h_ekf_i = load_pickle(hpath_ekf_i)
    h_ekf_e = load_pickle(hpath_ekf_e)
    h_ekf_fp = load_pickle(hpath_ekf_fp)
    print("Test Loading done")

    # Get variables out
    mmsl = expd['model']['state_length']
    fp_dict = expd['FPs']

    # 1. Plotting Ground Truth - Map + 2 Robots
    map_markers = {
        "label" : "map true",
        "marker" : "+",
        "markersize" : 10,
        "color" : "black",
        "linewidth" : 0
    }
    lm_map = LandmarkMap(map = np.asarray(expd["map"]["landmarks"]), workspace = expd["map"]["workspace"])
    lm_map.plot(**map_markers)       # plot true map
    r_dict = {
        "color" : "r",
        "label" : "r true",
        }
    plot_gt(h_mrekf,**r_dict);  # plot true path
    r2_dict = {
        "color" : "b",
        "label" : "r2 true"
    }
    plot_rs_gt(h_mrekf, **r2_dict)

    # 2. Plotting estimates
    # a. of Map: 
    marker_map_est = {
            "marker": "x",
            "markersize": 10,
            "color": "b",
            "linewidth": 0,
            "label" : "map est mr"
    }
    map_est_ell = {
        "color" : "b",
        "linestyle" : ":"
    }
    map_idcs_dyn = get_robot_idcs_map(h_mrekf)
    plot_map_est(h_mrekf, dynamic_map_idcs = map_idcs_dyn, state_length=mmsl, marker=marker_map_est, ellipse=map_est_ell)
    marker_inc = {
                "marker": "x",
                "markersize": 10,
                "color": "y",
                "linewidth": 0,
                "label" : "map est inc"
            }
    map_inc_ell ={
        "color": "y",
        "linestyle" : ":"
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
    plot_map_est(h_ekf_i, marker=marker_inc, ellipse = map_inc_ell)
    plot_map_est(h_ekf_e, marker=marker_exc)
    fp_map_idcs = get_fp_idcs_map(h_ekf_fp, list(fp_dict.values()))
    plot_map_est(h_ekf_fp, marker=marker_fp, dynamic_map_idcs=fp_map_idcs, state_length=mmsl)

    # b. of Paths
    r_est = {
        "color" : "r",
        "linestyle" : "-.",
        "label" : "r est"
    }
    # plot_xy_est(h_mrekf, **r_est)
    r2_est = {
        "color" : "b",
        "linestyle" : "dotted",
        "marker" : ".",
        "label" : "r2 est"
    }
    # plot_robs_est(h_mrekf, **r2_est)
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
    # plot_xy_est(h_ekf_e, **exc_r)
    # plot_xy_est(h_ekf_i, **inc_r)
    # plot_xy_est(h_ekf_fp, **fp_r)     
    
    # 3. Plotting Ellipses
    ## Plotting covariances
    covar_r_kws ={
        "color" : "r",
        "linestyle" : ":",
        "label" : "r covar"
    }
    plot_ellipse(h_mrekf, **covar_r_kws)
    for r in _get_robot_ids(h_mrekf):
        covar_r2_kws = {
                "color" : "b",
                "linestyle" : ":",
                "label" : "r{} covar".format(r)
            }
        plot_ellipse(h_mrekf, r, **covar_r2_kws)
    
    # ekf.plot_ellipse(**covar_r_kws);  # plot estimated covariance
    # ekf.plot_robot_estimates(**covar_r2_kws)
    # plt.show()

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
        # "linestyle" : ":",
        "label" : "lm {} fp covar".format(list(fp_dict.keys())[0])
    }
    plot_ellipse(h_ekf_e, **covar_exc_kws)
    plot_ellipse(h_ekf_i, **covar_inc_kws)
    covar_fp_kws["linestyle"] = ":"
    fp_list = list(fp_dict.values())
    for fp in fp_list:
        # TODO - this is where it breaks - does not work because the ellipse is not moving forward
        # plot_ellipse(h_ekf_fp, fp,  **covar_fp_kws)
        pass
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(os.path.join(rpath, rdir, 'test.jpg'), dpi=400)

    # displaying covariance
    # disp_P(h_mrekf)
    # plt.savefig(os.path.join(rpath, rdir, 'P.jpg'), dpi=400)

    # Evaluation section
    # Testing the Pnorms -> put into test section
    # Pnorm_hist = ekf.get_Pnorm()
    # lm_id_late = 7       # 7 only seen after a while
    # r_id = 0 + 100
    # t = 25
    # todo - all have been overwritten -> need to ensure that they still work.
    # print(get_Pnorm_lm(0))
    # print(get_Pnorm_lm(0, t))
    # print(get_Pnorm_lm(lm_id_late))
    # print(get_Pnorm_lm(lm_id_late, t))
    # ekf.get_Pnorm_r(r_id)
    # ekf.get_Pnorm_r(r_id, t)
   
    # Transform from map frame to the world frame -> now changed into three variables
    # calculating ate
    ate_exc = get_ATE(h_ekf_e, map_lms=lm_map)
    ate_inc = get_ATE(h_ekf_i, map_lms=lm_map)
    ekf_ate = get_ATE(h_mrekf, map_lms=lm_map, ignore_idcs=list(r2_dict.keys()))
    ate_fp = get_ATE(h_ekf_fp, map_lms=lm_map, ignore_idcs=fp_list)

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
    x_true = _get_xyt_true(h_mrekf)
    x_est =_get_robot_xyt_est(h_mrekf)
    x_inc = _get_robot_xyt_est(h_ekf_i)
    x_exc = _get_robot_xyt_est(h_ekf_e)
    x_fp = _get_robot_xyt_est(h_ekf_fp)
    dist_ekf = get_offset(x_true, x_est)
    dist_inc = get_offset(x_true, x_inc)
    dist_exc = get_offset(x_true, x_exc)
    dist_fp = get_offset(x_true, x_fp)

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