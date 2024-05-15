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
from mrekf.debug_utils import filter_dict, reload_from_exp
from mrekf.init_params import init_experiment
from mrekf.eval_utils import  get_ignore_idcs, get_transform, has_dynamic_lms,\
                            plot_gt, plot_xy_est, plot_dyn_gt, plot_dyn_est, plot_transformed_xy_est,\
                            get_transform_offsets, calculate_metrics, get_ATE, _get_xyt_true, get_dyn_idcs_map,\
                            plot_map_est

def parse_args(defdir : str):
    """
        Argument parser for the plot_case.py script
    """
        
    # quick settings
    parser = ArgumentParser(description="Helper script to rerun from a previously run experiment. Expects a hydra config dictionary and a GTLog in the directory")
    parser.add_argument("-d", "--directory", type=str, default=defdir, help="Name of the directory to load")
    # parser.add_argument("--debug", action="store_true", type=str, help="If given, will look for a <filters> subdirectory, load filters and compare")
    args = parser.parse_args()
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
        ate_d = calculate_metrics(simdict, ekf_hists, gt_hist, key)
        csv_row = pd.DataFrame(
            data=ate_d,
            index=[key]
        )
        # Writign to csv file
        with open(csv_f, 'a') as cf:
            csv_row.to_csv(cf, mode="a", header=cf.tell()==0)

    # append the new results to the csvfile-new, which should live in "directory"
    print("Test Debug line")


def inspect_csv(csvpath : str):
    df = pd.read_csv(csvpath, index_col=0)
    s1 = df["EKF_EXC-ate"] - df["EKF_MR:BF-ate"]
    df["ate_exc-mr:bf"] = s1
    s2 = df["EKF_MR:BF-ate"] - df["EKF_MR:SM-ate"]
    df["ate_mr:bf-mr:sm"] = s2
    s3 = df["EKF_MR:BF-ate"] - df["EKF_MR:KM-ate"]
    s1c = s1[s1 > 0. ].count()
    s2c = s2[s2 > 0 ].count() 
    s3c = s3[s3 > 0 ].count() 
    print("Percentage of cases where MRBF is lower than Exclusive: {:.2f}%".format((s1c / df.shape[0]) * 100))
    print("Percentage of cases where MRBF is lower than SM: {:.2f}%".format((s2c / df.shape[0]) * 100))
    print("Percentage of cases where MRBF is lower than KM {:.2f}%".format((s3c / df.shape[0]) * 100))
    print("Means of ATES:\nEKF_EXC:{}\nEKF_INC:{}\nMR:BF {}\nMR:KM {}\nMR:SM {}\n".format(
        df["EKF_EXC-ate"].mean(), df["EKF_INC-ate"].mean(), df["EKF_MR:BF-ate"].mean(), df["EKF_MR:KM-ate"].mean(), df["EKF_MR:SM-ate"].mean() 
    ))
    print("STDs of ATES:\nEKF_EXC:{}\nEKF_INC:{}\nMR:BF {}\nMR:KM {}\nMR:SM {}\n".format(
        df["EKF_EXC-ate"].std(), df["EKF_INC-ate"].std(), df["EKF_MR:BF-ate"].std(), df["EKF_MR:KM-ate"].std(), df["EKF_MR:SM-ate"].std() 
    ))
    print("MAX of ATES:\nEKF_EXC:{}\nEKF_INC:{}\nMR:BF {}\nMR:KM {}\nMR:SM {}\n".format(
        df["EKF_EXC-ate"].max(), df["EKF_INC-ate"].max(), df["EKF_MR:BF-ate"].max(), df["EKF_MR:KM-ate"].max(), df["EKF_MR:SM-ate"].max() 
    ))
    print("MIN of ATES:\nEKF_EXC:{}\nEKF_INC:{}\nMR:BF {}\nMR:KM {}\nMR:SM {}\n".format(
        df["EKF_EXC-ate"].min(), df["EKF_INC-ate"].min(), df["EKF_MR:BF-ate"].min(), df["EKF_MR:KM-ate"].min(), df["EKF_MR:SM-ate"].min() 
    ))
    print("Test debug line")

    # TODO: compare with FP filter!


if __name__=="__main__":
    pdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tmpdir = os.path.join(pdir, '.tmp')
    experiment_name = 'hydratest_20240515'
    casename = '2_1_3'
    defdir = os.path.join(tmpdir, experiment_name, casename)
    args = parse_args(defdir)

    # Loading settings
    print(f"Reloading from: {args.directory}")
    cfg = reload_from_exp(args.directory)
    filtdir = os.path.join(args.directory, 'filters')
    simpath = os.path.join(args.directory, cfg.hydra.job.name + '_simdict.json')
    simdict = load_json(simpath)
    gt_hist = load_gt_from_dir(args.directory)

    # loading results
    csvf = os.path.join(tmpdir, experiment_name + ".csv")
    exp_res = load_exp_from_csv(csvf, casename)    
    
    # loading filters
    if not os.path.isdir(filtdir):
        print(f"{filtdir} does not exist. Running again.")
        nfilts = init_experiment(cfg)
        for filt in nfilts:
            filt.rerun_from_hist(gt_hist)
        ekf_hists = {ekf.description : ekf.history for ekf in nfilts}
    else:
        ekf_hists = load_histories_from_dir(filtdir)

    # plotting
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
        "linewidth" : 0,
    }
    # Splitting the histories and settings
    ekf_hist_1 = filter_dict(ekf_hists, *["MR:SM"])
    ekf_hist_2 = filter_dict(ekf_hists, *["DATMO:SM"])
    ekf_hist_exc = filter_dict(ekf_hists, *["EXC"])
    ekf_hist_inc = filter_dict(ekf_hists, *["INC"])
    hist_subd = [ekf_hist_1, ekf_hist_2, ekf_hist_inc, ekf_hist_exc]
    # On each Subgraph
    # Plot:
    # * true Map
    # * true Path
    # * legend
    # * Baselines?

    # create a mapping from key to axis object? "BF" in k -> axs[0,0]
    cfg_h_dict = {}
    # for each in the subdicts, do an sca and plot
    for i, ax in enumerate(axs.ravel()):
        plt.sca(ax)

        # Plot Truth basics
        lm_map.plot(**map_markers);       # plot true map

        r_dict = {
            "color" : "black",
            "label" : "r true"
            }
        plot_gt(hist=gt_hist, **r_dict)
        r_dict["color"] = "b"
        r_dict["label"] = "r2 true"
        plot_dyn_gt(hist_gt=gt_hist, **r_dict)
        
        # plot special stuff
        if i >= len(hist_subd): break
        plot_dict = hist_subd[i]
        
        for k, hist in plot_dict.items():
            cfg = simdict[k]
            cfg_h_dict[k] = (cfg, hist)

            # getting the transform
            ign_idcs = get_ignore_idcs(cfg, simdict)
            tf = get_transform(hist, map_lms=lm_map, ignore_idcs=ign_idcs)

            # Plot map estimates
            marker = {
                "marker" : "x",
                "label" : "map est {}".format(k)
            }
            plot_map_est(hist, cfg, marker=marker)

            marker["label"] += " tf"
            plot_map_est(hist, cfg, tf=tf, marker=marker)

            # Plotting path estimates
            r_est = {
                # "color" : "r",
                "linestyle" : "-.",
                "label" : "r est: {}".format(k)
            }
            plot_xy_est(hist, **r_est)

            # plot transformed estimates
            t_d, R_d = get_transform_offsets(tf, angle=True)
            r_est["label"] = "r est tf {}".format(k)
            # print("Estimated transforms for: {}\nFrom calc:\nt:\n{},\theta:\n{}\ndistances:\nt\n{}\ntheta\n{}\nFrom csv:\n\tt:{},\n\ttheta:{}".format(k,
                    # tf[:2], tf[2], t_d, R_d, exp_res[f"{k}-translation_dist"], exp_res[f"{k}-rotation_dist"]))
            x_true = _get_xyt_true(gt_hist)
            ate_est, _ =  get_ATE(
                hist = hist,
                map_lms = lm_map,
                x_t = x_true,
                ignore_idcs = ign_idcs 
                )
            print("ATE for {} from csv: {:.5f} \t from calc: {:.5f}".format(
                k, exp_res[f"{k}-ate"], np.sqrt(ate_est.mean())
            ))
            plot_transformed_xy_est(hist, tf, **r_est)
            if has_dynamic_lms(cfg):
                r2_est = {
                    # "color" : "b",
                    "linestyle" : "dotted",
                    "marker" : ".",
                }
                plot_dyn_est(hist, cfg, **r2_est)
                plot_dyn_est(hist, cfg, tf=tf, **r2_est)
            
            plt.title(k)
            plt.legend()
        plt.xlim((-12, 12))
        plt.ylim((-12, 12))
    plt.suptitle("Seed: {}    Static: {}    Dynamic: {}".format(
            simdict['seed'], simdict['map']['num_lms'], len(simdict['dynamic'])
        ))        
    plt.tight_layout()
    plt.show()            
    print("Test Debug line")