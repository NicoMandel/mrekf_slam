"""

"""
import os.path
import numpy as np
import matplotlib.pyplot as plt

from math import pi

# own import
from mrekf.utils import load_json, load_histories_from_dir
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
    
    ##############################
    # PLOTTING Experiment
    ##############################
    
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
    plot_gt(hist=h_gt, **r_dict)
    r_dict["color"] = "b"
    r_dict["label"] = "r2 true"
    plot_rs_gt(hist=h_gt, **r_dict)
    marker_map_est = map_markers
    marker_map_est["color"] = "b"
    marker_map_est["label"] = "map est mr"
    marker_map_est["marker"] = "x"
    map_est_ell = {
        "color" : "b",
        "linestyle" : ":"
    }
    map_idcs_dyn = get_robot_idcs_map(simdict["Dynamic True"], h_mrekf)
    
    
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