"""
    Util files to help with evaluation of experiments.
    Only working on histories loaded from pickle files
    ! function to calculate the map distance of each lm to the best-case (exc)? After alignment?
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
    ks = list(hist[-1].robotsx.keys())
    return ks

def plot_gt(hist, *args, block=None, **kwargs):
    xyt = _get_xyt_true(hist)
    plt.plot(xyt[:, 0], xyt[:, 1], *args, **kwargs)
    if block is not None:
        plt.show(block=block)

def plot_rs_gt(hist, *args, block=None, rids : list = None, **kwargs):
    hd = _get_robots_xyt(hist, rids)
    for k, v in hd.items():
        kwargs["label"] = "rob: {}".format(k)
        plt.plot(v[:,0], v[:,1], *args, **kwargs)
    if block is not None:
        plt.show(block=block)

def get_robot_idx(hist, r_id : int):
    return hist[-1].seen_robots[r_id][2]

def _get_robot_idcs(hist) -> int:
    ks = list([v[2] for _, v in hist[-1].seen_robots.items()])
    return ks

def get_robot_idcs_map(hist) -> int:
    ks = _get_robot_idcs(hist)
    nk = [n - 3 for n in ks]
    return nk

def _get_fp_idcs(hist, fp_list):
    # TODO - this doesn't seem right...
    ks = list([v[2] for k, v in hist[-1].landmarks.items() if k in fp_list])
    return ks

def get_fp_idcs_map(hist, fp_list) -> int:
    ks = _get_fp_idcs(hist, fp_list)
    nk = [n -3 for n in ks]
    return nk

def get_idx_start_t(hist, idx : int) -> int:
    start_t = None
    for i, h in enumerate(hist):
        if len(h.xest) > idx:
            start_t = i
            break
    return start_t

def _split_states(x, P, r_idxs, state_length : int):
    b_x = np.ones(x.shape, dtype=bool)
    r_state_len = state_length
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


def plot_map_est(hist, marker=None, ellipse=None, confidence=0.95, block=None, dynamic_map_idcs : list = None, state_length : int = None):
    """
        Function to plot the map estimates positions of static landmarks. If dynamic_map_idcs is None, will plot the last estimate of all map markers
        Will exclude all dynamic_map_idcs from the plotting.
    """
    xest = hist[-1].xest[3:]
    Pest = hist[-1].Pest[3:,3:]
    if dynamic_map_idcs is not None:
        xest, Pest, _, _ = _split_states(xest, Pest, dynamic_map_idcs, state_length)
    xest = xest.reshape((-1,2))
    
    _plot_map_est(xest, Pest, marker=marker, ellipse=ellipse, confidence=confidence, block=block)

def plot_xy_est(hist, **kwargs):
    xyt = np.array([h.xest[:3] for h in hist])
    _plot_xy_est(xyt, **kwargs)

def plot_robs_est(hist, rob_id = None, **kwargs):
    if rob_id is None:
        ids = _get_robot_ids(hist)
        for rid in ids:
            ridx = get_robot_idx(hist, rid)
            st = get_idx_start_t(hist, ridx)
            xyt = np.array([h.xest[ridx : ridx + 2] for h in hist[st:]])
            kwargs["label"] = "rob: {} est".format(rid)
            _plot_xy_est(xyt, **kwargs)
    else:
        rob_idx = get_robot_idx(hist)
        start_t = get_idx_start_t(hist, rob_idx)
        xyt = np.array(hist[start_t:].xest[rob_idx:rob_idx+2]) 
        _plot_xy_est(xyt, **kwargs)

def _plot_xy_est(xyt, **kwargs):
    """
        Function to plot xy estimates of the robot.
    """
    plt.plot(xyt[:,0], xyt[:,1], **kwargs)