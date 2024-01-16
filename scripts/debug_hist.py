import os.path
from argparse import ArgumentParser

from mrekf.debug_utils import find_experiments, load_experiment, compare_histories

def parse_args(tmpdir):

    # quick settings
    parser = ArgumentParser(description="Scrip to debug ")
    parser.add_argument("-d", "--directory", help="Directory where files be read from. Defaults to .tmp", type=str, default=tmpdir)
    parser.add_argument("-e", "--exp", action="append", help="what the experiments are named that should be compared. read into a list")
    args = vars(parser.parse_args())
    return args


if __name__=="__main__":
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    tmpdir = os.path.join(basedir, '.tmp')
    args = parse_args(tmpdir)

    outdir = args["directory"]
    if not args["exp"]:
        # if is none, read all .json in experiments
        exps = find_experiments(args["directory"])
    else:
        exps = args["exp"]
    
    expdict = {}
    for exp in exps:
        expdict[exp] = load_experiment(outdir, exp)

    old_exp = expdict['old'][0]
    old_gt_h = expdict["old"][1]['GT']
    old_exc_h = expdict["old"][1]['EKF_EXC']

    new_exp = expdict['new'][0]
    new_gt_h = expdict["new"][1]['GT']
    new_exc_h = expdict["new"][1]['EKF_EXC']
    compare_histories(old_exc_h, new_exc_h)
    print("Test Debug line")
