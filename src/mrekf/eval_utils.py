"""
    Util files to help with evaluation of experiments.
    Only working on histories loaded from pickle files
    ! function to calculate the map distance of each lm to the best-case (exc)? After alignment?
"""
import numpy as np
import matplotlib.pyplot as plt
from spatialmath import base

from mrekf.ekf_base import EKF_base
from roboticstoolbox.mobile import LandmarkMap

def _get_xyt_true(hist) -> np.ndarray: 
    xyt = [v.xtrue for v in hist]
    return np.asarray(xyt)

def _get_xyt_est(hist) -> list:
    xyt = [v.xest for v in hist]
    return xyt

def _get_P(hist) -> list:
    P = [v.Pest for v in hist]
    return P

def _get_robot_P(hist) -> list:
    P = [v.Pest[:2, :2] for v in hist]
    return P

def _get_robot_xyt_est(hist) -> list:
    xyt = [v.xest[:2] for v in hist]
    return xyt

def _get_robots_xyt(hist, rids = None) -> dict:
    """
        Getting the true path of the robots
    """
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

def _get_robots_xyt_est(hist, start_t : int, ind : int) -> list:
    l = [h.xest[ind : ind +2] for h in hist[start_t:]]
    return l

def get_robots_xyt_est(hist, r_id : int) -> list:
    r_ind = get_robot_idx(hist, r_id)
    r_start = get_idx_start_t(hist, r_ind)
    if r_start == None:
        print("Robot {} never found in map!".format(r_id))
    xyt = _get_robots_xyt_est(hist, r_start, r_ind)
    return xyt

def _get_robot_P_est(hist, start_t : int, ind : int) -> list:
    P = [h.Pest[ind: ind+2, ind : ind+2] for h in hist[start_t:]]
    return P

def get_robot_P_est(hist, r_id : int) -> list:
    r_ind = get_robot_idx(hist, r_id)
    r_start = get_idx_start_t(hist, r_ind)
    if r_start == None:
        print("Robot {} never found in map!".format(r_id))
    P = _get_robot_P_est(hist, r_start, r_ind)
    return P

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

def plot_ellipse(hist, rob_id : int = None, confidence=0.95, N=10, block=None, **kwargs):
    if rob_id is None:
        # Get the xyt of the robot
        xyt = _get_robot_xyt_est(hist)
        P = _get_robot_P(hist)
    else:
        # get the robot index in the map
        xyt = get_robots_xyt_est(hist, rob_id)
        P = get_robot_P_est(hist, rob_id)
        # put that into the function _plot_ellipse
    _plot_ellipse(xyt, P, confidence, N, block, **kwargs)

def _plot_ellipse(xyt : np.ndarray, Pt : np.ndarray, confidence=0.95, N=10, block=None, **kwargs):
    """
        Function to plot ellipses. xyt and Pt have to be cleaned already
    """
    assert Pt[0].shape == (2,2), "Pt not cleaned. Please ensure that only the right indices are given"
    assert len(xyt) == len(Pt), "Length of P and xyt not equal. Double Check"
    
    nhist = len(xyt)
    if "label" in kwargs:
        label = kwargs["label"]
        del kwargs["label"]
    else:
        label = f"{confidence*100:.3g}% confidence"        
    for k in np.linspace(0, nhist - 1, N):
        k = round(k)
        x_loc = xyt[k]
        P_loc = Pt[k]
        if k == 0:
            base.plot_ellipse(
                P_loc,
                centre=x_loc,
                confidence=confidence,
                label=label,
                inverted=True,
                **kwargs,
            )
        else:
            base.plot_ellipse(
                P_loc,
                centre=x_loc,
                confidence=confidence,
                inverted=True,
                **kwargs,
            )
    if block is not None:
        plt.show(block=block)

def disp_P(hist, t : int = -1, colorbar=False):
    P = _get_P(hist)
    P = P[t]
    z = np.log10(abs(P))
    mn = min(z[~np.isinf(z)])
    z[np.isinf(z)] = mn
    plt.xlabel("State")
    plt.ylabel("State")

    plt.imshow(z, cmap="Reds")
    if colorbar is True:
        plt.colorbar(label="log covariance")
    elif isinstance(colorbar, dict):
        plt.colorbar(**colorbar)

def get_Pnorm(hist, k=None, ind=None, sl :int = 2):
    """
    Get covariance norm from simulation

    :param k: timestep, defaults to None
    :type k: int, optional
    :return: covariance matrix norm
    :rtype: float or ndarray(n)

    If ``k`` is given return covariance norm from simulation timestep ``k``, else
    return all covariance norms as a 1D NumPy array.

    :seealso: :meth:`get_P` :meth:`run` :meth:`history`
    """
    if k is not None:
        P = hist[k].Pest
        if ind is not None:
            s_ind = _get_state_idx(ind)
            P = P[s_ind : s_ind + sl, s_ind : s_ind + sl]
        return np.sqrt(np.linalg.det(P))
    else:
        P = [h.Pest for h in hist]
        if ind is not None:
            s_ind = _get_state_idx(ind)
            P = [p[s_ind : s_ind + sl, s_ind : s_ind + sl] for p in P]   
        p = [np.sqrt(np.linalg.det(x)) for x in P]
        return np.array(p)
    

def get_transform(hist, map_lms : LandmarkMap, ignore_idcs : list) -> tuple[np.array, np.ndarray, float]:
        """
        directly from PC - slight modification of get transformation params
        Transformation from estimated map to true map frame

        :param map: known landmark positions
        :type map: :class:`LandmarkMap`
        :return: transform from ``map`` to estimated map frame
        :rtype: SE2 instance

        Uses a least squares technique to find the transform between the
        landmark is world frame and the estimated landmarks in the SLAM
        reference frame.

        :seealso: :func:`~spatialmath.base.transforms2d.points2tr2`
        """
        p = []
        q = []

        for lm_id in self._landmarks.keys():
            if lm_id > 99: continue     # case when we have a robot in the map - do not use it for alingment
            p.append(map_lms[lm_id])
            q.append(self.landmark_x(lm_id))

        p = np.array(p).T
        q = np.array(q).T

        return EKF_base.get_transformation_params(q, p)
    
def get_ATE(hist, map_lms : LandmarkMap, t : slice = None) -> np.ndarray:
    """
        Function to get the absolute trajectory error
        uses the staticmethod calculate_ATE
        if t is given, uses slice of t
    """

    x_t = _get_xyt_true(hist)
    x_e = _get_xyt_est(hist)

    if t is not None:
        x_t = x_t[:,t]
        x_e = x_e[:,t]

    # getting the transform parameters
    c, Q, s = get_transform(hist, map_lms)

    return EKF_base.calculate_ATE(x_t, x_e, s, Q, c)