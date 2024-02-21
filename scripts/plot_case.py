"""
    File to plot a specific case
"""
import os.path
from argparse import ArgumentParser
from mrekf.utils import load_json, load_exp_from_csv, load_histories_from_dir, load_gt_from_dir

def parse_args(defdir : str):
    """
        Argument parser for the simple_inherit.py script
    """
    default_case = "20240216_170318"         # currently chosen default case, where the EKF_MR:BF is terrible
    defexp = "debug_2_true_vals"

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

    print("Test Debug line")