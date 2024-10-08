import os.path
from argparse import ArgumentParser

from mrekf.utils import reload_from_exp, load_histories_from_dir, load_gt_from_dir
from mrekf.debug_utils import _compare_filter_and_new

def parse_args(defdir : str):
    """
        Argument parser for the reruncript
    """

    # quick settings
    parser = ArgumentParser(description="Helper script to rerun from a previously run experiment. Expects a hydra config dictionary and a GTLog in the directory")
    parser.add_argument("-d", "--directory", type=str, default=defdir, help="Name of the directory to load")
    # parser.add_argument("--debug", action="store_true", type=str, help="If given, will look for a <filters> subdirectory, load filters and compare")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    pdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    defdir = os.path.join(pdir, '.tmp', 'fullxlsx_20240710', '2_1_0')
    args = parse_args(defdir)

    print(f"Reloading from: {args.directory}")
    cfg = reload_from_exp(args.directory)

    # Optional - load the old filter directories
    # filtdir = os.path.join(args.directory, 'filters')
    # ekf_hists = load_histories_from_dir(filtdir)
    gt_hist = load_gt_from_dir(args.directory)

    # _compare_filter_and_new(ekf_histdict=ekf_hists, cfg=cfg, gt_hist=gt_hist)
    
    # Setup a filter here

    # Rerun through the observations
    for h in gt_hist:
        print("Run filter with inputs:\n\todo:\t{}\nand\n\tz:\t{}".format(h.odo, h.z))
        print("Compare filter outputs with true values:\nrobot state:\t{}\ndynamic:\t{}".format(h.xtrue, h.robotsx))
