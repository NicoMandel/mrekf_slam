import os.path
from argparse import ArgumentParser

import numpy as np
from spatialmath import SE2, SO2

from mrekf.utils import load_histories_from_dir, load_gt_from_dir, load_json, load_exp_from_csv
from mrekf.eval_utils import fromto, tofrom

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

    # quick settings
    parser = ArgumentParser(description="file to plot a specific case")
    parser.add_argument("-n", "--name", type=str, default=default_case, help="Name of the file / folder combination to look for in the input directory")
    parser.add_argument("-d", "--directory", type=str, default=defdir, help="Name of the default directory to look for")
    parser.add_argument("-e", "--experiment", default=defexp, type=str, help="Name of the experiment, named like the csv file where to look for the case")
    args = vars(parser.parse_args())
    return args

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

    # debug the histories.
    datmo_hist = ekf_hists["EKF_DATMO:SM"]
    for i, gt in enumerate(gt_hist):
        t = gt.t
        dm = datmo_hist[i]
        
        # Estimated and true values
        x_r_est : np.ndarray = dm.xest[:3]
        x_r : np.ndarray = gt.xtrue

        x_i_w : np.ndarray = gt.robotsx[100]
        x_i_est : np.ndarray = dm.trackers[100].xest
        # x_i_est_w : np.ndarray = fromto(x_r_est, x_i_est)

        # running with spatialmath
        se_r_est = SE2(x_r_est)
        se_r = SE2(x_r)

        # true position of the dyn lm in the robot frame
        x_i_r : np.ndarray = se_r.inv() * x_i_w[:2]

        # estimated position of the dyn lm in the robot frame:
        x_i_w_est = se_r_est * x_i_est 

        print("Positions in the robot frame:\nEstimated:{}\nTrue:{}".format(
            x_i_est, x_i_r
        ))
        print("Positions in the world frame:\nEstimated:{}\nTrue:{}".format(
            x_i_w_est, x_i_w
        ))

        x_i_est_w_se = se_r_est.inv() * x_i_est
        x_i_test = se_r_est * x_i_est

        print("True position of r_0 in world frame: \n{}\n\
              Estimated position of r_0 in world frame:\n{}\n\n\
              True position of r_100 in world frame: \n{}\n\
              Estimated position of r_100 in world frame:\n{}\n".format(
                x_r, x_r_est,
                x_i_w, x_i_test
              ))
        print(80*"=")
        # values for recalculating
        odo = gt.odo
        z : dict = gt.z
        print("Test Debug line")
