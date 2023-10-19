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

def _get_r_xyt_est(hist):
    xyt = [v.xest[:3] for v in hist]
    return xyt

def _get_P(hist) -> list:
    P = [v.Pest for v in hist]
    return P

def _get_robot_P(hist) -> list:
    P = [v.Pest[:2, :2] for v in hist]
    return P

def _get_robot_xyt_est(hist) -> np.ndarray:
    xyt = [v.xest[:2] for v in hist]
    return np.asarray(xyt)

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

def _get_lm_ids(hist) -> list:
    ks = list(hist[-1].landmarks.keys())
    return ks

def _get_lm_idx(hist, lm_id : int) -> list:
    return hist[-1].landmarks[lm_id][2]

def get_lm_xye(hist, lm_id : int) -> np.ndarray:
    lm_idx = _get_lm_idx(hist, lm_id)
    xye = hist[-1].xest[lm_idx : lm_idx + 2]
    return xye

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

def _get_robot_idcs(ekf_d : dict, hist) -> int:
    """
        Get the indices in the state vector of the landmarks that are considered dynamic BY THE ROBOT
    """
    dyn_lm_list = get_dyn_lm(ekf_d)
    if dyn_lm_list is not None:
        ks = list([v[2] for k, v in hist[-1].landmarks.items() if k in dyn_lm_list])
    # ks = list([v[2] for _, v in hist[-1].seen_robots.items()])
    else:
        ks = []
    return ks

def get_dyn_lm(ekf_d : dict) -> list:
    """
        Get a list of the dynamic landmarks from an ekf dictionary.
        If the key does not exist, returns None - can be tested for existence
    """
    return ekf_d.get("dynamic_lms")

def get_robot_idcs_map(ekf_d : dict, hist) -> int:
    """
        Get the indices of the robot that are considered dynamic
    """
    ks = _get_robot_idcs(ekf_d, hist)
    nk = [n - 3 for n in ks]
    return nk


def get_start_t(hist, lm_id : int):
    """
        Function to get a start time when a landmark apperas in the history
    """
    start_t = 0
    lm_ind = _get_lm_idx(lm_id)
    for i, h in enumerate(hist):
        if len(h.xest > lm_ind):
            start_t = i
            break
    return start_t


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
    if dynamic_map_idcs:
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

### newly inserted stuff here    
### section on static methods for calculating the offset - to get the ATE
def get_transformation_params(p1 : np.ndarray, p2 : np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
        function from PC transforms2d -> with changes according to J. Skinner's PhD Thesis!
        [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/spatialmath/base/transforms2d.py]] -> points2tr2
        from spatialmath.base.transforms2d import points2tr2
        Function to get the transformation parameters between a true map and an estimated map
        to be used with the ATE calculation.
        p2 are the TRUE coordinates
        p1 are the ESTIMATED coordinates
        depends on the map alignment.
        scale will most likely always be something close to 1
        is an instance of ICP
        Chapter 2.5.4 in Thesis from John Skinner
    """
    p1_centr = np.mean(p1, axis=1)
    p2_centr = np.mean(p2, axis=1)

    p1_centered = p1 - p1_centr[:, np.newaxis]
    p2_centered = p2 - p2_centr[:, np.newaxis]

    # computing moment matrix
    M = np.dot(p2_centered, p1_centered.T)

    # svd composition on M
    U, _, Vt = np.linalg.svd(M)

    # rotation between PCLs
    s = [1, np.linalg.det(U) * np.linalg.det(Vt)]
    R = U @ np.diag(s) @  Vt

    # This is where we differ from PC. we estimate scale by:
    scale = (p2_centered * (R @ p1_centered)).sum() /  np.sum(p2_centered**2)
    # translation - also different from PC, according to sJS
    t = p2_centr - scale * (R @ p1_centr)

    return t, R, scale

def calculate_ATE(x_true : np.ndarray, x_est : np.ndarray, s : float, Q : np.ndarray, c : np.ndarray) -> np.ndarray:
    """
        function to calculate the ATE according to John Skinner Chapter 2.5.3.2
        except for the mean(). that can be done afterwards.
        ignore the rotation component in the differences between the trajectories. 
        We do not care in this case!
    """
    val = x_true[:,:2] - s * (Q @ x_est[:,:2].T).T
    # alt
    # val = x_true[:,:2] - s * (x_est[:,:2] @ Q)
    val += c
    return val**2    

def get_transform(hist, map_lms : LandmarkMap, ignore_idcs : list = []) -> tuple[np.array, np.ndarray, float]:
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

        for lm_id in _get_lm_ids(hist):
            if lm_id in ignore_idcs: continue   # function to skip robots
            p.append(map_lms[lm_id])
            xe = get_lm_xye(hist, lm_id)
            q.append(xe)

        p = np.array(p).T
        q = np.array(q).T

        return get_transformation_params(q, p)
    
def get_ATE(hist, map_lms : LandmarkMap, x_t : np.ndarray = None, t : slice = None, ignore_idcs : list = []) -> np.ndarray:
    """
        Function to get the absolute trajectory error
        uses the staticmethod calculate_ATE
        if t is given, uses slice of t
    """
    if x_t is None:
        x_t = _get_xyt_true(hist)
    x_e = _get_r_xyt_est(hist)

    if t is not None:
        x_t = x_t[:,t]
        x_e = x_e[:,t]

    # getting the transform parameters
    c, Q, s = get_transform(hist, map_lms, ignore_idcs)

    return calculate_ATE(x_t, np.asarray(x_e), s, Q, c)

def get_offset(x_true : np.ndarray, x_est : np.ndarray) -> np.ndarray:
        """
            function to get the distance using the true values
            ! careful -> because of dynamic objects, we get a scale and rotation factor that is not considered
            have to be better with ATE
            ! ignores angular differences
        """
        x_diff = (x_true[:,:2] - x_est[:,:2])**2
        # theta_diff = base.angdiff(x_true[:,2], x_est[:,2])
        return x_diff

def compare_update(h1, h2, t : slice = None) -> np.ndarray:
        """
            TODO - taken from EKF_base - remove there!
            Function to compare the update in the x_est step for the robot by looking at the K @ v part of the equation for the state update step
            if the values are larger, there's a larger update happening
        """
        K1_h = [h.K for h in h1]
        K2_h = [h.K for h in h2]

        in1_h = [h.innov for h in h1]
        in2_h = [h.innov for h in h2]

        if t is not None:
            K1_h = K1_h[:,t]
            K2_h = K2_h[:,t]

            in1_h = in1_h[:,t]
            in2_h = in2_h[:,t]
        
        assert len(K1_h) == len(in1_h), "Length of innovation and Kalman Gain for first history are not the same. Please double check"
        assert len(K2_h) == len(in2_h), "Length of innovation and Kalman Gain for second history are not the same. Please double check"
        assert len(in1_h) == len(in2_h), "Length of innovations between histories is not the same. Please double check"

        
        u1 = [k1h @ in1_h[i] for i, k1h in enumerate(K1_h)]
        u2 = [k2h @ in2_h[i] for i, k2h in enumerate(K2_h)]

        # u1[:3] are now the updates for the first robot
        # u2[:3] are now the updates for the second robot
        return False

### Section on Evaluation
# TODO - section taken from EKF_Base
def get_Pnorm(self, k=None):
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
        return np.sqrt(np.linalg.det(self._history[k].Pest))
    else:
        p = [np.sqrt(np.linalg.det(h.Pest)) for h in self._history]
        return np.array(p)

def _filler_func(self, dim : int) -> np.ndarray:
    return np.sqrt(np.linalg.det(-1 * np.ones((dim, dim))))

def _ind_in_P(self, P : np.ndarray, m_ind : int) -> bool:
    return True if P.shape[0] > m_ind else False

def get_Pnorm_map(self, map_ind : int, t : int = None, offset : int  = 2):
    if t is not None:
        P_h = self.history[t].Pest
        P = P_h[map_ind : map_ind + offset, map_ind : map_ind + offset] if self._ind_in_P(P_h, map_ind) else self._filler_func(offset)
        return np.sqrt(np.linalg.det(P))
    else:
        p = [np.sqrt(np.linalg.det(h.Pest[map_ind : map_ind + offset, map_ind : map_ind + offset])) if self._ind_in_P(h.Pest, map_ind) else self._filler_func(offset) for h in self._history]
        return np.array(p)

def get_Pnorm(self, lm_id : int, t : int  = None):
    ind = self.landmark_index(lm_id)
    return self.get_Pnorm_map(ind, t)
    