"""
    Command line utility for the simple_inherit.py script in the same folder. to be used as outside function when running experiments
"""
from argparse import ArgumentParser
import os
from datetime import datetime, date
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=10000, edgeitems=30)
import pandas as pd
from roboticstoolbox import LandmarkMap

from mrekf.utils import read_config, dump_json, dump_gt, dump_ekf
from mrekf.run import run_simulation
from mrekf.eval_utils import get_ignore_idcs, get_transform

import matplotlib.pyplot as plt
import warnings
import matplotlib
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
from mrekf.eval_utils import plot_xy_est, plot_map_est, plot_dyn_gt, plot_gt, plot_ellipse, get_dyn_lms, get_dyn_idcs_map, plot_dyn_est,\
has_dynamic_lms, plot_transformed_xy_est, calculate_metrics


from omegaconf import DictConfig, OmegaConf
import hydra

CONFDIR = os.path.abspath(os.path.join(os.path.basename(__file__), '..', 'config'))

@hydra.main(version_base=None, config_path=CONFDIR, config_name="config")
def main(cfg : DictConfig) -> None:
    """
        Hydra main function for accessing everything
    """
    # filepaths setup
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))

    print(OmegaConf.to_yaml(cfg))
    print(os.getcwd())
    print(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    simdict, gt_hist, ekf_hists = run_simulation(cfg)
    ate_d = calculate_metrics(simdict, ekf_hists, gt_hist, outname) # TODO - continue here - outname is not defined

    # Turn into a pandas dataframe and append
    df = pd.DataFrame(
        data=ate_d,
        index=[outname]
    )
    csv_f = os.path.join(resultsdir, "{}.csv".format(args["csv"]))
    
    # Writign to csv file
    print("Appending to file: {}".format(csv_f))
    with open(csv_f, 'a') as cf:
        df.to_csv(cf, mode="a", header=cf.tell()==0)

    if cfg.debug:
        tmpdir = os.path.abspath(os.path.join(basedir, '.tmp'))
        debugdir = os.path.join(tmpdir, args["csv"])
        print("Debug flag activated. Writing all histories to {}".format(debugdir))
        os.makedirs(debugdir, exist_ok=True)
        outdir = os.path.join(debugdir, outname)
        jsonpath = os.path.join(debugdir, outname + '.json')
        dump_json(simdict, jsonpath)
        try:
            os.makedirs(outdir)
            dump_gt(gt_hist, outdir)
            for ekf_id, ekf_hist in ekf_hists.items():
                dump_ekf(ekf_hist, ekf_id, outdir)
        except FileExistsError:
            print("Folder {} already exists. Skipping".format(outdir))
    print("Test debug line")

def parse_args(confdir : str):
    """
        Argument parser for the simple_inherit.py script
    """
    conff = os.path.join(confdir, 'default_bf.yaml')

    # quick settings
    parser = ArgumentParser(description="Wrapper script to run experiments for MR-EKF simulations")
    parser.add_argument("-o", "--output", help="Directory where files be output to. If none, will just run and append to csv", type=str, default=None)
    parser.add_argument("-d", "--dynamic", type=int, default=3, help="Number of dynamic landmarks to use")
    parser.add_argument("-s", "--static", type=int, default=3, help="Number of static landmarks to use")
    parser.add_argument("-c", "--csv", default="default", type=str, help="Name of the csv file to be written into the <results> directory. Defaults to <default>")
    
    # longer settings
    parser.add_argument("--config", help="Location of the config .yaml file to be used for the experiments. If None given, takes default with body frame model from config folder.", default=conff)
    parser.add_argument("--seed", type=int, help="Which seed to use. Defaults to 1", default=1)
    parser.add_argument("--offset", type=int, default=100, help="The offset for the ids for the dynamic landmarks. Defaults to 100.")
    parser.add_argument("--workspace", type=int, default=10, help="Workspace size for the map.")    
    parser.add_argument("--time", type=int, default=60, help="Simulation time to run in seconds.")
    parser.add_argument("--plot", action="store_true", help="Set this to show a plot at the end")
    parser.add_argument("--debug", action="store_true", help="If set, will debug to .tmp in the base folder")
    parser.add_argument("--true", action="store_true", help="Whether to use the true value for hidden states when inserting, instead of guessed values")

    # Disabling filters
    parser.add_argument("--incfilter", action="store_false", help="If set, will not run the inclusive filter (false negative)")
    parser.add_argument("--fpfilter", action="store_false", help="If set, will not run the false positive filter (false positive)")
    parser.add_argument("--dynamicfilter", action="store_false", help="If set, will not run the dynamic filter (true positive)")
    parser.add_argument("--datmo", action="store_false", help="If set, will not run the DATMO implementation (baseline)")
    args = vars(parser.parse_args())
    return args

if __name__=="__main__":
    main()

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

    # returns dictionaries of hists. those can be used to plot or calculate ATE
    simdict, gt_hist, ekf_hists = run_simulation(args, cd)

    ate_d = calculate_metrics(simdict, ekf_hists, gt_hist, outname)

    # Turn into a pandas dataframe and append
    df = pd.DataFrame(
        data=ate_d,
        index=[outname]
    )
    # print(df)
    if args["debug"]:   
        tmpdir = os.path.abspath(os.path.join(basedir, '.tmp'))
        debugdir = os.path.join(tmpdir, args["csv"])
        print("Debug flag activated. Writing all histories to {}".format(debugdir))
        os.makedirs(debugdir, exist_ok=True)
    csv_f = os.path.join(resultsdir, "{}.csv".format(args["csv"]))
    
    # Writign to csv file
    print("Appending to file: {}".format(csv_f))
    with open(csv_f, 'a') as cf:
        df.to_csv(cf, mode="a", header=cf.tell()==0)
        

    if args["debug"]:
        outdir = os.path.join(debugdir, outname)
        jsonpath = os.path.join(debugdir, outname + '.json')
        dump_json(simdict, jsonpath)
        try:
            os.makedirs(outdir)
            dump_gt(gt_hist, outdir)
            for ekf_id, ekf_hist in ekf_hists.items():
                dump_ekf(ekf_hist, ekf_id, outdir)
        except FileExistsError:
            print("Folder {} already exists. Skipping".format(outdir))
    if args["plot"]:
        plt.figure(figsize=(16,10))

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
        lm_map.plot(**map_markers);       # plot true map

        r_dict = {
            "color" : "black",
            "label" : "r true"
            }
        plot_gt(hist=gt_hist, **r_dict)
        r_dict["color"] = "b"
        r_dict["label"] = "r2 true"
        plot_dyn_gt(hist_gt=gt_hist, **r_dict)
        
        # Splitting the histories and settings
        cfg_h_dict = {}
        for k, hist in ekf_hists.items():
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
            tf = get_transform(hist, map_lms=lm_map, ignore_idcs=ign_idcs)
            r_est["label"] = "r est tf {}".format(k)
            plot_transformed_xy_est(hist, tf, **r_est)
            if has_dynamic_lms(cfg):
                r2_est = {
                    # "color" : "b",
                    "linestyle" : "dotted",
                    "marker" : ".",
                    "label" : "r2 est {}".format(k)
                }
                plot_dyn_est(hist, cfg, **r2_est)
                plot_dyn_est(hist, cfg, tf, **r2_est)
            
        plt.title("Seed: {}    Static: {}    Dynamic: {}".format(
            simdict['seed'], simdict['map']['num_lms'], len(simdict['dynamic'])
        ))
        plt.legend()
        plt.show()            
        
        ##############################
        # PLOTTING Experiment
        ##############################
        

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