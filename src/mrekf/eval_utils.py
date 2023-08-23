"""
    Util files to help with evaluation of experiments.
    Only working on histories loaded from pickle files
"""
import numpy as np
import matplotlib.pyplot as plt
from spatialmath import base

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

def _split_states(x, P, robot_ids):
    # TODO: figure out where the robot idcs are in the map! FUCK -> store "seen robots" and "robot_map_index" ?
    r_idxs = [self.robot_index(r_id) - 3 for r_id in robot_ids]
    b_x = np.ones(x.shape, dtype=bool)
    r_state_len = self.motion_model.state_length
    for r_idx in r_idxs:
        b_x[r_idx : r_idx + r_state_len] = False
    x_stat = x[b_x]
    P_stat = P[np.ix_(b_x, b_x)]        
    x_dyn = x[~b_x]
    P_dyn = P[np.ix_(~b_x, ~b_x)]

    return x_stat, P_stat, x_dyn, P_dyn

def _plot_map_est(x, P, marker=None, ellipse=None, confidence=0.95, block=None):
    plt.plot(x[:, 0], x[:, 1], **marker)
    if ellipse is not None:
        for i in range(x.shape[0]):
            Pi = P[i : i + 2, i : i + 2]
            # todo change this -> not correct ellipses
            # put ellipse in the legend only once
            if i == 0:
                base.plot_ellipse(
                    Pi,
                    centre=x[i, :],
                    confidence=confidence,
                    inverted=True,
                    label=f"{confidence*100:.3g}% confidence",
                    **ellipse,
                )
            else:
                base.plot_ellipse(
                    Pi,
                    centre=x[i, :],
                    confidence=confidence,
                    inverted=True,
                    **ellipse,
                )
    # plot_ellipse( P * chi2inv_rtb(opt.confidence, 2), xf, args{:});
    if block is not None:
        plt.show(block=block)


def plot_map_est(hist, marker=None, ellipse=None, confidence=0.95, block=None, dynamic : bool = False):
    xest = hist[-1].xest[3:]
    Pest = hist[-1].Pest[3:,3:]
    if dynamic:
        xest, Pest, _, _ = _split_states(xest, Pest, _get_robot_ids(hist))
        xest = xest.reshape((-1,2))
    
    _plot_map_est(xest, Pest, marker=marker, ellipse=ellipse, confidence=confidence, block=block)
