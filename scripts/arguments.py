"""
    Command line utility for the simple_inherit.py script in the same folder. to be used as outside function when running experiments
"""
from argparse import ArgumentParser
import os
from datetime import datetime, date
import numpy as np
import pandas as pd
from roboticstoolbox import LandmarkMap
 
from mrekf.utils import read_config, dump_json, dump_gt, dump_ekf
from mrekf.run import run_simulation
from mrekf.eval_utils import _get_xyt_true, get_ignore_idcs, get_ATE

import matplotlib.pyplot as plt
from mrekf.eval_utils import plot_xy_est, plot_map_est, plot_dyn_gt, plot_gt, plot_ellipse, get_dyn_lms, get_dyn_idcs_map, plot_dyn_est, get_transform_offsets


def parse_args(confdir : str):
    """
        Argument parser for the simple_inherit.py script
    """
    conff = os.path.join(confdir, 'default.yaml')

    # quick settings
    parser = ArgumentParser(description="Wrapper script to run experiments for MR-EKF simulations")
    parser.add_argument("-o", "--output", help="Directory where files be output to. If none, will just run and append to csv", type=str, default=None)
    parser.add_argument("-d", "--dynamic", type=int, default=3, help="Number of dynamic landmarks to use")
    parser.add_argument("-s", "--static", type=int, default=3, help="Number of static landmarks to use")
    
    # longer settings
    parser.add_argument("--config", help="Location of the config .yaml file to be used for the experiments. If None given, takes default from config folder.", default=conff)
    parser.add_argument("--seed", type=int, help="Which seed to use. Defaults to 1", default=1)
    parser.add_argument("--offset", type=int, default=100, help="The offset for the ids for the dynamic landmarks. Defaults to 100.")
    parser.add_argument("--workspace", type=int, default=10, help="Workspace size for the map.")    
    parser.add_argument("--time", type=int, default=60, help="Simulation time to run in seconds.")
    parser.add_argument("--plot", action="store_true", help="Set this to show a plot at the end")
    args = vars(parser.parse_args())
    return args

if __name__=="__main__":
    # filepaths setup
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    resultsdir = os.path.join(basedir, 'results')
    confdir = os.path.join(basedir, 'config')
    args = parse_args(confdir)

    # read the configs from file
    cd = read_config(args['config'])

    # run script, pass the arguments as dictionary
    outname = datetime.today().strftime('%Y%m%d_%H%M%S')
    today = date.today().strftime("%Y%m%d")
    print("Test Debug line")

    # returns dictionaries of hists. those can be used to plot or calculate ATE
    simdict, gt_hist, ekf_hists = run_simulation(args, cd)

    # Calculate ATE
    workspace = np.array(simdict['map']['workspace'])
    mp = np.array(simdict['map']['landmarks'])
    lm_map = LandmarkMap(map=mp, workspace=workspace)
    x_true = _get_xyt_true(gt_hist)

    ate_d = {
        # "timestamp" : outname,
        "dynamic" : args["dynamic"],
        "static" : args["static"],
        "time" : args["time"],
        "seed" : args["seed"],
        "fp_count" : len(cd["fp_list"]),
        "motion_model" : simdict['EKF_MR']['motion_model']['type']
    }
    for ekf_id, ekf_hist in ekf_hists.items():
        cfg_ekf = simdict[ekf_id]
        ign_idcs = get_ignore_idcs(cfg_ekf, simdict)
        ate, (c, Q, s) =  get_ATE(
            hist = ekf_hist,
            map_lms = lm_map,
            x_t = x_true,
            ignore_idcs = ign_idcs 
            )
        
        ate_d[ekf_id + "-ate"] = ate.mean()

        # get transformation parameters
        c_d, Q_d, s_d  = get_transform_offsets(c, Q, s)
        ate_d[ekf_id + "-translation_dist"] = c_d
        ate_d[ekf_id + "-rotation_dist"] = Q_d
        ate_d[ekf_id + "-scale"] = s        
        
    
    # Turn into a pandas dataframe and append
    df = pd.DataFrame(
        data=ate_d,
        index=outname
    )
    print(df)

    csv_f = os.path.join(resultsdir, "ate_100.csv")
    with open(csv_f, 'a') as cf:
        df.to_csv(cf, mode="a", header=cf.tell()==0)
    simfpath = os.path.join(resultsdir, "configs", "100", outname + ".json")
    dump_json(simdict, simfpath)

    if args["output"]:
        outdir = os.path.join(resultsdir, outname)
        try:
            os.makedirs(outdir)
            dump_gt(gt_hist, outdir)
            for ekf_id, ekf_hist in ekf_hists.items():
                dump_ekf(ekf_hist, ekf_id, outdir)
        except FileExistsError:
            print("Folder {} already exists. Skipping".format(outdir))
    elif args["plot"]:
        plt.figure(figsize=(16,10))

        # Splitting the histories and settings
        h_ekfmr = ekf_hists["EKF_MR"]
        h_ekfinc = ekf_hists["EKF_INC"]
        h_ekfexc = ekf_hists["EKF_EXC"]
        h_ekffp = ekf_hists["EKF_FP"]

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
        # plt.savefig(loc, dpi=400)
        plt.show()    