"""
    Command line utility for the simple_inherit.py script in the same folder. to be used as outside function when running experiments
"""
from argparse import ArgumentParser
import os.path
from datetime import date, datetime
from mrekf.utils import read_config
from mrekf.run import run_simulation

def parse_args():
    """
        Argument parser for the simple_inherit.py script
    """
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    confdir = os.path.join(basedir, 'config')
    conff = os.path.join(confdir, 'default.yaml')

    # quick settings
    parser = ArgumentParser(description="Wrapper script to run experiments for MR-EKF simulations")
    parser.add_argument("-o", "--output", help="Directory where files be output to. If none, will plot run. Directory will contain timestamped subdir", type=str, default=None)
    parser.add_argument("-d", "--dynamic", type=int, default=1, help="Number of dynamic landmarks to use")
    parser.add_argument("-s", "--static", type=int, default=20, help="Number of static landmarks to use")

    # longer settings
    parser.add_argument("--config", help="Location of the config .yaml file to be used for the experiments. If None given, takes default from config folder.", default=conff)
    parser.add_argument("--seed", type=int, help="Which seed to use. Defaults to 1", default=1)
    parser.add_argument("--offset", type=int, default=100, help="The offset for the ids for the dynamic landmarks. Defaults to 100")
    
    args = vars(parser.parse_args())
    return args

if __name__=="__main__":
    args = parse_args()

    # read the configs from file
    cd = read_config(args['config'])

    # run script, pass the arguments as dictionary
    now = datetime.now()
    tim = "{}-{}-{}".format(now.hour, now.minute, now.second)
    outname = f"{date.today()}-{tim}"
    print("Test Debug line")

    # returns dictionaries of hists. those can be used to plot or calculate ATE
    simdict, hists = run_simulation(args, cd)

    # what to do with the returns
    if args["output"]:
        dump_json(simdict, simfpath)
        dump_gt(sim, rdir)
        for ekf in ekf_list:
            dump_ekf(ekf, rdir)
