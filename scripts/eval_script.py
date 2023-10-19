"""

"""
import os.path
import numpy as np
import matplotlib.pyplot as plt

from roboticstoolbox import LandmarkMap

# own import
from mrekf.utils import load_json, load_histories_from_dir, load_gt_from_dir
from mrekf.eval_utils import plot_gt, plot_rs_gt, get_robot_idcs_map, plot_map_est, plot_ellipse, _get_robot_ids, \
get_fp_idcs_map, plot_robs_est, plot_xy_est

if __name__=="__main__":
   
    
    ############################# 
    # LOADING Experiment
    #############################
    bdir = os.path.dirname(__file__)
    pdir = os.path.abspath(os.path.join(bdir, '..'))
    rdir = os.path.join(pdir, 'results', "inherit")

    # load a dictionary out from the simulation configs
    simfpath = os.path.join(rdir, 'config.json')
    simdict = load_json(simfpath)

    # loading histories
    ekf_hists = load_histories_from_dir(rdir)
    gt_hist = load_gt_from_dir(rdir)
    
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
        "color" : "r",
        "label" : "r true"
        }
    plot_gt(hist=gt_hist, **r_dict)
    r_dict["color"] = "b"
    r_dict["label"] = "r2 true"
    plot_rs_gt(hist=gt_hist, **r_dict)

    # Splitting the histories and settings
    h_ekfmr = ekf_hists["EKF_MR"]
    h_ekfinc = ekf_hists["EKF_INC"]
    h_ekfexc = ekf_hists["EKF_EXC"]
    h_ekffp = ekf_hists["EKF_FP"]

    cfg_ekfmr = simdict["EKF_MR"]
    cfg_ekfinc = simdict["EKF_INC"]
    cfg_ekfexc = simdict["EKF_EXC"]
    cfg_ekffp = simdict["EKF_FP"]

    # Plotting the Map estimates
    marker_map_est = map_markers
    marker_map_est["color"] = "b"
    marker_map_est["label"] = "map est mr"
    marker_map_est["marker"] = "x"
    map_est_ell = {
        "color" : "b",
        "linestyle" : ":"
    }
    map_idcs_dyn = get_robot_idcs_map(cfg_ekfmr , h_ekfmr)
    ekf_mr_mmsl = cfg_ekfmr["motion_model"]["state_length"]   
    plot_map_est(h_ekfmr, dynamic_map_idcs = map_idcs_dyn, state_length=ekf_mr_mmsl, marker=marker_map_est, ellipse=map_est_ell)
    marker_map_est["color"] = "y"
    marker_map_est["label"] = "map est inc"
    map_est_ell["color"] = "y"
    plot_map_est(h_ekfinc, marker=marker_map_est, ellipse = map_est_ell)
    marker_map_est["color"] = map_est_ell["color"] = "g"
    marker_map_est["label"] = "map est exc"
    plot_map_est(h_ekfexc, marker=marker_map_est)
    marker_map_est["color"] = map_est_ell["color"] = "m"
    marker_map_est["label"] = "map est fp"
    ekf_fp_mmsl = cfg_ekffp["motion_model"]["state_length"]
    fp_map_idcs = get_robot_idcs_map(cfg_ekffp, h_ekffp)
    plot_map_est(h_ekffp, marker=marker_map_est, dynamic_map_idcs=fp_map_idcs, state_length=ekf_fp_mmsl, ellipse=map_est_ell)
    plt.legend()
    plt.show()
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
    plot_robs_est(h_ekfmr, **r2_est)
    r2_list = _get_robot_ids(h_ekfmr) 
    for r in r2_list:
        covar_r2_kws["label"] = "r{} covar".format(r)
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
    plot_xy_est(h_ekf_fp, **r_est)     
    plot_ellipse(h_ekf_fp, **covar_r_kws)
    plt.legend()
    plt.show()    