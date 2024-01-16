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