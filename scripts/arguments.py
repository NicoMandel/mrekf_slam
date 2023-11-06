"""
    Command line utility for the simple_inherit.py script in the same folder. to be used as outside function when running experiments
"""
from argparse import ArgumentParser
import os.path

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
    parser.add_argument("-o", "--output", help="Directory where files be output to. If none, will plot run", type=str, default=None)
    parser.add_argument("-d", "--dynamic", type=int, default=1, help="Number of dynamic landmarks to use")
    parser.add_argument("-s", "--static", type=int, default=20, help="Number of static landmarks to use")

    # to put into CONFIG!
    parser.add_argument("-f", "--fp", type=int, default=1, action="append", help="Which landmarks to assume as false positives. Appends to a list")

    # longer settings
    parser.add_argument("--config", help="Location of the config .yaml file to be used for the experiments. If None given, takes default from config folder.", default=conff)
    parser.add_argument("--seed", type=int, help="Which seed to use. Defaults to 1", default=1)
    parser.add_argument("--offset", type=int, default=100, help="The offset for the ids for the dynamic landmarks. Defaults to 100")
    
    args = vars(parser.parse_args())
    return args

if __name__=="__main__":
    args = parse_args()

    # read the configs from file
        
    # run script, pass the arguments as dictionary
    
