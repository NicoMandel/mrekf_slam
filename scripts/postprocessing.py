import os.path
import numpy as np
from mrekf.utils import load_json, load_pickle
from roboticstoolbox import LandmarkMap
from mrekf.eval_utils import plot_gt, plot_rs_gt, plot_map_est, get_robot_idcs_map, get_fp_idcs_map, \
plot_xy_est, plot_robs_est, plot_ellipse, _get_robot_ids, disp_P, \
get_ATE, _get_xyt_true, get_offset, _get_robot_xyt_est, _get_r_xyt_est
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
    fp_list = list(fp_dict.keys())

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
    r_dict["color"] = "b"
    r_dict["label"] = "r2 true"
    plot_rs_gt(h_mrekf, **r_dict)

    # 2. Plotting estimates
    # a. of Map: 
    marker_map_est = map_markers
    marker_map_est["color"] = "b"
    marker_map_est["label"] = "map est mr"
    marker_map_est["marker"] = "x"
    map_est_ell = {
        "color" : "b",
        "linestyle" : ":"
    }
    map_idcs_dyn = get_robot_idcs_map(h_mrekf)
    plot_map_est(h_mrekf, dynamic_map_idcs = map_idcs_dyn, state_length=mmsl, marker=marker_map_est, ellipse=map_est_ell)
    marker_map_est["color"] = "y"
    marker_map_est["label"] = "map est inc"
    map_est_ell["color"] = "y"
    plot_map_est(h_ekf_i, marker=marker_map_est, ellipse = map_est_ell)
    marker_map_est["color"] = map_est_ell["color"] = "g"
    marker_map_est["label"] = "map est exc"
    plot_map_est(h_ekf_e, marker=marker_map_est)
    marker_map_est["color"] = map_est_ell["color"] = "m"
    marker_map_est["label"] = "map est fp"
    fp_map_idcs = get_fp_idcs_map(h_ekf_fp, list(fp_dict.values()))
    plot_map_est(h_ekf_fp, marker=marker_map_est, dynamic_map_idcs=fp_map_idcs, state_length=mmsl, ellipse=map_est_ell)

    # b. of Paths
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
    for r in _get_robot_ids(h_mrekf):
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

    # inspecting the estimated robots variables over time:
    # r_index = ekf.robot_index(list(ekf.seen_robots.keys())[0])
    # state_len = mot_model.state_length
    # r_list = np.array([h.xest[r_index : r_index + state_len] for h in ekf.history if len(h.xest) > r_index])
    # plt.figure()
    # plt.plot(r_list[:,0], label="x")
    # plt.plot(r_list[:,1], label="y")
    # plt.plot(r_list[:,2], label="v")
    # plt.plot(r_list[:,3], label="theta")
    # plt.legend()
    # plt.show()
   
    # Transform from map frame to the world frame -> now changed into three variables
    # calculating ate
    rids = _get_robot_ids(h_mrekf)
    x_true = _get_xyt_true(h_mrekf)
    # x_true = _get_r_xyt_est(h_mrekf)
    ate_exc, _ = get_ATE(h_ekf_e, map_lms=lm_map, x_t=x_true)
    ate_inc, _ = get_ATE(h_ekf_i, map_lms=lm_map, x_t=x_true, ignore_idcs=rids)
    ekf_ate, _ = get_ATE(h_mrekf, map_lms=lm_map, x_t=x_true, ignore_idcs=rids)
    ate_fp, _ = get_ATE(h_ekf_fp, map_lms=lm_map, x_t=x_true, ignore_idcs=fp_list)

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