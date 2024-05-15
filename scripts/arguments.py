"""
    Command line utility for the simple_inherit.py script in the same folder. to be used as outside function when running experiments
"""
from argparse import ArgumentParser
import os
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=10000, edgeitems=30)
import pandas as pd

from mrekf.utils import dump_json, dump_gt, dump_ekf
from mrekf.run import run_simulation
from mrekf.debug_utils import compare_cfg_dictionaries

import warnings
import matplotlib
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
from mrekf.eval_utils import calculate_metrics


from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig

CONFDIR = os.path.abspath(os.path.join(os.path.basename(__file__), '..', 'config'))

@hydra.main(version_base=None, config_path=CONFDIR, config_name="config")
def main(cfg : DictConfig) -> None:
    """
        Hydra main function for accessing everything
    """
    # filepaths setup
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    resultsdir = os.path.join(basedir, '.tmp')
    
    hydraconf = HydraConfig.get()
    outdir = hydraconf.runtime.output_dir
    jobname = hydraconf.job.name
    simdict, gt_hist, ekf_hists = run_simulation(cfg)

    # Store Ground Truth history first for reprocessing
    dump_gt(gt_hist, outdir)
    # Storing experiment settings -> simdict
    jsonpath = os.path.join(outdir, jobname + '_simdict.json')
    dump_json(simdict, jsonpath)

    # Calculate metrics
    ate_d = calculate_metrics(simdict, ekf_hists, gt_hist)    
    # Turn into a pandas dataframe and append
    df = pd.DataFrame(
        data=ate_d,
        index=[jobname]
    )
    csv_f = os.path.join(resultsdir, "{}.csv".format(cfg.experiment_name))
    # Writign to csv file
    print("Appending to file: {}".format(csv_f))
    with open(csv_f, 'a') as cf:
        df.to_csv(cf, mode="a", header=cf.tell()==0)

    # Debug - compare the histories
    if cfg.debug:
        compare_cfg_dictionaries(outdir, cfg)

    # Store Hist activated - also store filter histories    
    if cfg.store_hist:
        hist_dir = os.path.join(outdir, "filters")
        print("Store histories flag activated. Writing all histories to {}".format(hist_dir))

        try:
            os.makedirs(hist_dir)
            for ekf_id, ekf_hist in ekf_hists.items():
                dump_ekf(ekf_hist, ekf_id, hist_dir)
        except FileExistsError:
            print("Folder {} already exists. Skipping".format(hist_dir))
    print("Finished experiment for: {} static landmarks, {} dynamic landmarks with seed {}".format(cfg.static, cfg.dynamic, cfg.seed))

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

    # For plotting, see file ./scripts/plot_case.p