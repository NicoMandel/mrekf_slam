"""
    File to plot a specific case
"""
import os
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from roboticstoolbox import LandmarkMap

from mrekf.utils import load_json, load_exp_from_csv, load_histories_from_dir, load_gt_from_dir
from mrekf.eval_utils import  get_ignore_idcs, get_transform, has_dynamic_lms,\
                            plot_gt, plot_xy_est, plot_dyn_gt, plot_dyn_est, plot_transformed_xy_est,\
                            get_transform_offsets, _get_xyt_true, get_ATE

def parse_args(defdir : str):
    """
        Argument parser for the simple_inherit.py script
    """
    default_case = "20240216_170318"         # currently chosen default case, where the EKF_MR:BF is terrible
    default_case="20240216_170320"
    defexp = "debug_2_true_vals"

    # quick settings
    parser = ArgumentParser(description="file to plot a specific case")
    parser.add_argument("-n", "--name", type=str, default=default_case, help="Name of the file / folder combination to look for in the input directory")
    parser.add_argument("-d", "--directory", type=str, default=defdir, help="Name of the default directory to look for")
    parser.add_argument("-e", "--experiment", default=defexp, type=str, help="Name of the experiment, named like the csv file where to look for the case")
    args = vars(parser.parse_args())
    return args

def recalculate(directory : str, experiment : str, csvfn : str):
    pdir = os.path.join(directory, experiment)
    # for every subdirectory in the pdir
    list_subfolders_with_paths = [f.name for f in os.scandir(pdir) if f.is_dir()]
    csv_f = os.path.join(directory, csvfn)
    # load the json
    for key in tqdm(list_subfolders_with_paths):
        jsonf = os.path.join(pdir, key + ".json")
        simdict = load_json(jsonf)

        # load the experiments
        rdir = os.path.join(pdir, key)
        ekf_hists = load_histories_from_dir(rdir)
        gt_hist = load_gt_from_dir(rdir)
        csv_row_df = recalc_case(simdict, ekf_hists, gt_hist, key)

        # Writign to csv file
        with open(csv_f, 'a') as cf:
            csv_row_df.to_csv(cf, mode="a", header=cf.tell()==0)

    # append the new results to the csvfile-new, which should live in "directory"
    print("Test Debug line")

def recalc_case(simdict : dict, ekf_hists : dict, gt_hist : dict, identity : str) -> pd.DataFrame:
    # Calculate ATE
    workspace = np.array(simdict['map']['workspace'])
    mp = np.array(simdict['map']['landmarks'])
    lm_map = LandmarkMap(map=mp, workspace=workspace)
    x_true = _get_xyt_true(gt_hist)

    ate_d = {
        # "timestamp" : outname,
        "dynamic" : len(simdict["dynamic"]),
        "static" : lm_map._nlandmarks,
        "time" : len(gt_hist) / 10,             # ! kinda - hardcoded
        "seed" : simdict["seed"],
        "fp_count" : 1,                             # ! hardcoded!
        # "motion_model" : simdict['motion_model']['type']
    }
    for ekf_id, ekf_hist in ekf_hists.items():
        cfg_ekf = simdict[ekf_id]
        ign_idcs = get_ignore_idcs(cfg_ekf, simdict)
        ate, R_e, t_e =  get_ATE(
            hist = ekf_hist,
            map_lms = lm_map,
            x_t = x_true,
            ignore_idcs = ign_idcs 
            )
        
        ate_d[ekf_id + "-ate"] = np.sqrt(ate.mean())

        # get transformation parameters
        t_d, R_d  = get_transform_offsets(t_e, R_e)
        ate_d[ekf_id + "-translation_dist"] = t_d
        ate_d[ekf_id + "-rotation_dist"] = R_d
    df = pd.DataFrame(
        data=ate_d,
        index=[identity]
    )
    return df

def filter_dict(in_dict : dict, *inkey : list) -> dict:
    return {k:v for ik in inkey for k,v in in_dict.items() if ik in k}

if __name__=="__main__":
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tmpdir = os.path.join(basedir, '.tmp')
    args = parse_args(tmpdir)

    directory = args["directory"]
    experiment = args["experiment"]
    name = args["name"]

    recalculate(directory, experiment, "newfile.csv")

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

    # TODO: plot filter values on specific axis object
    # TODO: use plt.sca(ax) for plt.gca() acts 
    f, axs = plt.subplots(2,2, figsize=(16,10))

    # Plotting the True Map and robot states.
    workspace = np.array(simdict['map']['workspace'])
    mp = np.array(simdict['map']['landmarks'])
    lm_map = LandmarkMap(map=mp, workspace=workspace)
    # Plotting Ground Truth
    map_markers = {
        "label" : "map true",
        "marker" : "+",
        "markersize" : 10,
        "color" : "k",
        "linewidth" : 0
    }
    # Splitting the histories and settings
    cfg_h_dict = {}
    ekf_hist_subdict = filter_dict(ekf_hists, *["MR"])
    ekf_hist_baselines = filter_dict(ekf_hists, *["INC", "EXC"])

    # On each Subgraph
    # Plot:
    # * true Map
    # * true Path
    # * legend
    # * Baselines?

    # create a mapping from key to axis object? "BF" in k -> axs[0,0]
    
    for k, hist in ekf_hist_subdict.items():
        if "SM" in k:
            plt.sca(axs[0,1])
        if "KM" in k:
            plt.sca(axs[1,0])
        if "BF" in k:
            plt.sca(axs[1,1])
        else: continue
        cfg = simdict[k]
        cfg_h_dict[k] = (cfg, hist)

        # Plotting path estimates
        r_est = {
            # "color" : "r",
            "linestyle" : "-.",
            "label" : "r est: {}".format(k)
        }
        plot_xy_est(hist, **r_est)

        # plot transformed estimates
        ign_idcs = get_ignore_idcs(cfg, simdict)
        t_e, R_e = get_transform(hist, map_lms=lm_map, ignore_idcs=ign_idcs)
        t_d, R_d = get_transform_offsets(t_e, R_e)
        r_est["label"] = "r est tf {}".format(k)
        print("Estimated transforms for: {}\nFrom calc:\nt:\n{},\nR:\n{}\ndistances:\nt\n{}\nR\n{}\nFrom csv:\n\tt:{},\n\tR:{}".format(k,
                t_e, R_e, t_d, R_d, exp_res[f"{k}-translation_dist"], exp_res[f"{k}-rotation_dist"]))
        plot_transformed_xy_est(hist, R_e, t_e, **r_est)
        if has_dynamic_lms(cfg):
            r2_est = {
                # "color" : "b",
                "linestyle" : "dotted",
                "marker" : ".",
                "label" : "r2 est {}".format(k)
            }
            plot_dyn_est(hist, cfg, **r2_est)
            plot_dyn_est(hist, cfg, transform=(R_e, t_e), **r2_est)
    
    for ax in axs.ravel():
        plt.sca(ax)
        lm_map.plot(**map_markers);       # plot true map

        r_dict = {
            "color" : "black",
            "label" : "r true"
            }
        plot_gt(hist=gt_hist, **r_dict)
        r_dict["color"] = "b"
        r_dict["label"] = "r2 true"
        plot_dyn_gt(hist_gt=gt_hist, **r_dict)
        plt.title("Seed: {}    Static: {}    Dynamic: {}".format(
            simdict['seed'], simdict['map']['num_lms'], len(simdict['dynamic'])
        ))
        plt.legend()
    plt.show()            
    print("Test Debug line")