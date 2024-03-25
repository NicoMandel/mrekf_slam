import os.path
from pathlib import Path
from mrekf.utils import load_histories_from_dir, load_gt_from_dir, load_json

def find_experiments(dirn : str) -> list:
    """
        returns a list of all .json paths in the experiment
    """
    dn = Path(dirn)
    fs = dn.glob("*.json")
    assert fs, "Path {} is empty. Check again".format(dirn)
    return list([f.stem for f in fs])

def load_experiment(dir : str, exp : str) -> tuple[dict, dict]:
    """
        function to load a single experiment.
        Returns Json and dict of all histories 
    """
    try:
        p = os.path.join(dir, exp)
        gt_h = {"GT": load_gt_from_dir(p)}
        h_d = load_histories_from_dir(p)
        jsp = os.path.join(dir, exp + ".json")
        jsd = load_json(jsp)
    except FileNotFoundError as e:
        raise e("Path {} does not appear to exist".format(p))

    ds = {**gt_h, **h_d}
    return jsd, ds

def compare_histories(h1, h2, t: slice = None):
    """

    """
    if t is not None:
        h1 = h1[t]
        h2 = h2[t]
    
    h1d = {h.t : h._asdict() for h in h1}
    h2d = {h.t : h._asdict() for h in h2}

    for k, v in h1d.items():
        for vk, vv in v.items():
            if vv != h2d[k][vk]:
                raise Exception("histories differ at time: {}, value: {} are different: check: \n{} and {}".format(k, vk, vv, h2d[k][vk]))
        if v != h2d[k]:
            print("Differs at time:")

def filter_dict(in_dict : dict, *inkey : list) -> dict:
    return {k:v for ik in inkey for k,v in in_dict.items() if ik in k}