import os.path
from pathlib import Path
from deepdiff import DeepDiff
from copy import deepcopy
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mrekf.ekf_base import BasicEKF
from mrekf.datmo import DATMO
from mrekf.utils import load_histories_from_dir, load_gt_from_dir, load_json, reload_from_exp
from mrekf.init_params import init_experiment

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

def isdatmo(hist) -> bool:
    return hasattr(hist, 'trackers')

def _check_history_consistency(gt_hist : list, ekf_list : list):

    for ekf in ekf_list:
        ekf : BasicEKF
        nekf = deepcopy(ekf)
        nekf.rerun_from_hist(gt_hist)
        _check_xest(ekf.history, nekf.history)
        assert np.array_equal(nekf.history[-1].xest, ekf.history[-1].xest), f"X estimate at last time {gt_hist[-1].t} not equal. Double check"
        assert np.array_equal(nekf.history[-1].Pest, ekf.history[-1].Pest), f"P at last time {gt_hist[-1].t} not equal. Double check"
    print("Histories for all filters are equal after clearing and rerunning")

def _check_xest(ekf_h, nekf_h) -> None | float:
    """
        checking if the estimated states are exactly equal when re-running the filter
    """

    for i, h1 in enumerate(ekf_h):
        h2 = nekf_h[i]
        x_en = h2.xest
        x_e = h1.xest
        if isdatmo(h1):
            t1 = h1.trackers
            t2 = h2.trackers
            try:
                __check_trackers_xest(t1, t2)
            except ValueError:
                print(f"at time {h1.t}")
        if not np.array_equal(x_en, x_e):
            return h1.t
        
    return None

def __check_trackers_xest(t1 : dict, t2 : dict):
    for k, tl1 in t1.items():
        tl2 = t2[k]
        if not np.array_equal(tl1.xest, tl2.xest):
            print(f"Not equal for tracker {k}: orig {tl1.xest}\n new:{tl2.xest}")
            raise ValueError
        
def compare_cfg_dictionaries(experiment_dir : str, cfg : DictConfig):
    """  
        difference between a hydraconf reloaded from a directory and the original hydraconf
    """
    
    hydraconf_reload = reload_from_exp(experiment_dir=experiment_dir)
    d = DeepDiff(hydraconf_reload, cfg)
    print("Difference between the Config files: ")
    print(d)

def filthist_equal(filt : BasicEKF, hist) -> bool:
    """
        Function to check whether a filter and a history belong together
    """
    return True if filt.description == hist[0].description else False

def hist_equal(h1, h2) -> bool:
    """
        Function to check whether two histories are for the same filter type
    """
    return True if h1[0].description == h2[0].description else False

def _compare_filter_and_new(ekf_histdict : dict, cfg : DictConfig, gt_hist):
    """
        Function to compare a filter coming out of a simulation directly with a filter that is newly initialized.
        Has to stay here, otherwise circular import dependency with debug_utils.py
    """
    nfilts = init_experiment(cfg)
    for filt in nfilts:
        
        # finding the corresponding history
        for k, of_h in ekf_histdict.items():
            if filthist_equal(filt, of_h):
                old_h = of_h
                break
                
        filt.description += "_n"
        filt.rerun_from_hist(gt_hist)
        check_t =_check_xest(old_h, filt.history)
        if check_t is not None:
            print(f"Found divergence for filter: {filt.description} at time: {check_t}")
        print(f"New Filter: {filt.description} and old filter: {old_h[0].description} have equal x_est all the way through")
    print("Checked all values.")