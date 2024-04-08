import os.path
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

import spatialmath.base as smb
from spatialmath import SE2

from mrekf.utils import load_histories_from_dir, load_gt_from_dir, load_json, load_exp_from_csv
from mrekf.debug_utils import filter_dict
from mrekf.transforms import pol2cart, cart2pol, inverse, forward, dist
from mrekf.ekf_base import TRACKERLOG, GT_LOG

def parse_args(defdir : str):
    """
        Argument parser for the simple_inherit.py script
    """
    # default_case="20240216_170320"
    # default_case="20240301_140950"
    # default_case="20240304_115745"
    # default_case="20240304_125400"
    defexp = "datmo_test_20"
    defexp = "datmo_test_3"

    default_case = "20240304_155441"
    defexp = "datmo_test_2"

    # for new datmo cases
    default_case = "20240325_143819"
    defexp = "datmo_baseline_interesting"

    default_case = "20240325_143916"

    # For cases testing the sign
    default_case = "20240325_163716"    # negative case
    default_case = "20240325_163657"    # positive case
    defexp = "test_sign"

    defexp = "someweirdtest"
    default_case = "20240403_140601"

    # quick settings
    parser = ArgumentParser(description="file to plot a specific case")
    parser.add_argument("-n", "--name", type=str, default=default_case, help="Name of the file / folder combination to look for in the input directory")
    parser.add_argument("-d", "--directory", type=str, default=defdir, help="Name of the default directory to look for")
    parser.add_argument("-e", "--experiment", default=defexp, type=str, help="Name of the experiment, named like the csv file where to look for the case")
    args = vars(parser.parse_args())
    return args

def find_max_datmo(datmo_hist : dict) -> tuple[int, float]:
    """
        Function to find the index when the datmo estimate diverges the most
    """
    x_est_prev = None
    t = 0
    d = 0.0

    for i, dm in enumerate(datmo_hist):
        # setup work
        x_i_est : np.ndarray = dm.trackers[100].xest[:2]
        if x_est_prev is None:
            x_est_prev = x_i_est.copy()
            continue
        
        dd = dist(x_i_est, x_est_prev)
        if dd > d:
            t = i
            d = dd
        
        x_est_prev = x_i_est.copy()
        
    print("Maximum distance: {} at time: {}".format(d, t))
    return t, d

def filter_from_hist(gt_hist : list):
    from mrekf.datmo import DATMO
    from mrekf.dynamic_ekf import Dynamic_EKF
    from mrekf.ekf_base import BasicEKF
    from mrekf.sensor import SensorModel
    from mrekf.motionmodels import KinematicModel

    from roboticstoolbox import Bicycle
    from copy import deepcopy

    Vr = np.array([[0.2, 0.0],[0.0, np.deg2rad(5)]]) ** 2
    x0r = np.array([0., 0., np.deg2rad(5)])
    # rtype = configs["vehicle_model"]["type"]
    robot = Bicycle(covar=Vr, x0=x0r, 
            animation="car")
    
    W = np.array([[0.4, 0.0], [0.0, np.deg2rad(5)]]) ** 2
    sensor = SensorModel(robot, lm_map=None, covar=W)

    mm = KinematicModel(
        dt=robot.dt,
        V=deepcopy(Vr)
    )

    datmo_kwargs = {
        "description": "datmo_FromHist",
        "dynamic_ids" : [100],
        "robot" : (robot, Vr),
        "motion_model" : mm,
        "sensor" : (sensor, W),
        "x0" : deepcopy(x0r),
        "P0" : np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, np.deg2rad(10)]])
    }

    datmo_filter = DATMO.from_gt_hist(gt_hist, **datmo_kwargs)

    base_kwargs = deepcopy(datmo_kwargs)
    base_kwargs["ignore_ids"] = [100]
    del base_kwargs["dynamic_ids"]
    del base_kwargs["motion_model"]
    base_ekf = BasicEKF.from_gt_hist(gt_hist, **base_kwargs)

    dyn_kwargs = deepcopy(datmo_kwargs)
    dyn_ekf = Dynamic_EKF.from_gt_hist(gt_hist, **dyn_kwargs)

    print("init true")


if __name__=="__main__":
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tmpdir = os.path.join(basedir, '.tmp')
    args = parse_args(tmpdir)

    directory = args["directory"]
    experiment = args["experiment"]
    name = args["name"]

    # loading experiment values from csv
    csvf = os.path.join(directory, experiment + ".csv")
    exp_res = load_exp_from_csv(csvf, name)
    
    # loading jsonfile for the experiment
    jsonf = os.path.join(args["directory"], experiment, name + ".json")
    simdict = load_json(jsonf)

    # Loading the histories
    rdir = os.path.join(directory,experiment, name)
    ekf_hists = load_histories_from_dir(rdir)
    gt_hist = load_gt_from_dir(rdir)

    filter_from_hist(gt_hist)

    # debug the histories.
    datmo_hist = ekf_hists["EKF_DATMO:SM"]
    ekf_hist_1 = filter_dict(ekf_hists, *["MR:BF"])
    ekf_hist_2 = filter_dict(ekf_hists, *["DATMO:BF"])
    ekf_hist_baselines = filter_dict(ekf_hists, *["INC", "EXC"])

    dm_hist = ekf_hist_2["EKF_DATMO:BF"]
    t, d = find_max_datmo(dm_hist)
    window = 3
    start_t = max(0, t-window)
    end_t = min(t + window, len(dm_hist))

    dm_snippet = dm_hist[start_t : end_t]
    gt_snippet = gt_hist[start_t : end_t]
    
    fig, ax = plt.subplots(1,1)
    for i, dms in enumerate(dm_snippet):
        gts : GT_LOG = gt_snippet[i]
        dms : TRACKERLOG

        # ax.clear()        

        # Estimated and true values
        T_r_est = SE2(dms.xest[:3])
        smb.trplot2(T_r_est.A, frame=f"est_{i}", color="r", width=0.2, ax=ax)

        T_r = SE2(gts.xtrue)
        smb.trplot2(T_r.A, frame=f"true_{i}", color="b", width=0.2, ax=ax)

        x_i : np.ndarray = gts.robotsx[100]
        # smb.plot_point(x_i[:2], "ys", text=f"p_{i}")
        smb.trplot2(SE2(x_i).A, frame=f"100_{i}", color="y", width=0.2, ax=ax)

        x_i_est : np.ndarray = dms.trackers[100].xest
        x_0_est : np.ndarray = forward(T_r_est.xyt(), x_i_est[:2])
        # smb.plot_point(x_0_est, "go", text=f"pe_{i}")
        
        theta_i_est = x_i_est[2] + dms.xest[2]  # ?
        T_i_est = SE2(x_0_est[0], x_0_est[1], theta_i_est)
        smb.trplot2(T_i_est.A, frame=f"100e_{i}", color="g", width=0.2, ax=ax)


        z = gts.z[100]

        xy_z = pol2cart(z)
        xy_z_k0e = forward(T_r_est.xyt(), xy_z)
        xy_z_k0t = forward(T_r.xyt(), xy_z)
        smb.plot_arrow(T_r_est.xyt()[:2], xy_z_k0e, color='k', linestyle=":", linewidth=0.8)
        # smb.plot_arrow(T_r.xyt()[:2], xy_z_k0t, color='k', linestyle=":", linewidth=0.8)
        ax.set_title("i: {:.2f}, max at: {}".format(gts.t, t/10))
        plt.show()
