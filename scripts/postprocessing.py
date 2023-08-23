import os.path
import numpy as np
from mrekf.utils import load_json, load_pickle
from roboticstoolbox import LandmarkMap
from mrekf.eval_utils import plot_gt, plot_rs
import matplotlib.pyplot as plt

if __name__=="__main__":
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

    # Plotting Ground Truth
    map_markers = {
        "label" : "map true",
        "marker" : "+",
        "markersize" : 10,
        "color" : "black",
        "linewidth" : 0
    }
    lm_map = LandmarkMap(map = np.asarray(expd["map"]["landmarks"]), workspace = expd["map"]["workspace"])
    lm_map.plot(**map_markers)       # plot true map
    # plt.show()
    r_dict = {
        "color" : "r",
        "label" : "r true",
        }
    plot_gt(h_mrekf,**r_dict);  # plot true path
    # Plot the second robot
    r2_dict = {
        "color" : "b",
        "label" : "r2 true"
    }
    plot_rs(h_mrekf, **r2_dict)
    plt.legend()
    plt.show()
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