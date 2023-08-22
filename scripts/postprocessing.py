import os.path
from mrekf.utils import load_experiment, load_history, load_pickle

if __name__=="__main__":
    fpath = os.path.dirname(__file__)
    bpath = os.path.abspath(os.path.join(fpath, '..'))
    
    rpath = os.path.join(bpath, "results")
    rdir = "1-BodyFrame-20-10"
    exp_path = os.path.join(rpath, rdir, rdir + ".json")
    hpath = os.path.join(rpath, rdir, "MREKFLog.pkl")

    expd = load_experiment(exp_path)
    # histd = load_history(hpath)
    histd = load_pickle(hpath)

    print("Test Run done")