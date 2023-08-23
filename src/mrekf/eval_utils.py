"""
    Util files to help with evaluation of experiments.
    Only working on histories loaded from pickle files
"""
import numpy as np
import matplotlib.pyplot as plt

def _get_xyt_true(hist) -> np.ndarray: 
    xyt = [v.xtrue for v in hist]
    return np.asarray(xyt)

def _get_xyt_est(hist) -> np.ndarray:
    xyt = [v.xest for v in hist]
    return np.asarray(xyt)

def _get_robots_xyt(hist, rids = None) -> dict:
    if rids is None:
        rids = _get_robot_ids(hist)
    hd = {k: np.asarray([h.robotsx[k] for h in hist]) for k in rids}
    return hd

def _get_robot_ids(hist) -> list:
    ks = hist[-1].robotsx.keys()
    return ks

def plot_gt(hist, *args, block=None, **kwargs):
    xyt = _get_xyt_true(hist)
    plt.plot(xyt[:, 0], xyt[:, 1], *args, **kwargs)
    if block is not None:
        plt.show(block=block)

def plot_rs(hist, *args, block=None, rids : list = None, **kwargs):
    hd = _get_robots_xyt(hist, rids)
    for k, v in hd.items():
        kwargs["label"] = "rob: {}".format(k)
        plt.plot(v[:,0], v[:,1], *args, **kwargs)
    if block is not None:
        plt.show(block=block)