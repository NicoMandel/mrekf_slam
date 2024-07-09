"""
    Util files to help with evaluation of experiments.
    Only working on histories loaded from pickle files
    ! function to calculate the map distance of each lm to the best-case (exc)? After alignment?
    TODO:
        * function to plot map and dynamic landmark estimate together
            * differentiate whether [dynamic_ids] is a key in the cfg dictionary 
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from spatialmath import base

from roboticstoolbox.mobile import LandmarkMap
from spatialmath import *

from mrekf.ekf_base import DATMOLOG
from mrekf.transforms import forward, inverse, get_transform_offsets, _get_angle, tf_from_tR, vect_dist

def isdatmo(hist) -> bool:
    """
        Function to test if a history is a DATMO object or normal EKF.
    """
    return isinstance(hist[0], DATMOLOG)

def _get_xyt_true(hist) -> np.ndarray:
    """
        !
        used with hist_gt
    """
    xyt = [v.xtrue for v in hist]
    return np.asarray(xyt)

def _get_r_xyt_est(hist) -> np.ndarray:
    xyt = [v.xest[:2] for v in hist]
    return np.asarray(xyt)

def get_state_est(hist, lm_id : int, offset : int = 2) -> dict:
    """
        :param state_ind: index of the 
        :param offset: how many states further to retrieve
    
        :return: state from time t -> start time of the index
        :rtype: dict
    """
    start_t = get_id_start_t(hist, lm_id)
    lm_idx = _get_lm_idx(hist, lm_id=lm_id)
    rd = {h.t : h.xest[lm_idx : lm_idx + offset] for h in hist[start_t:]}
    return rd

def _get_P(hist) -> list:
    P = [v.Pest for v in hist]
    return P

def _get_robot_P(hist) -> list:
    P = [v.Pest[:2, :2] for v in hist]
    return P

def _get_dyn_lm_xyt(hist_gt, rids : list = None) -> dict:
    """
        * used
        Getting the true path of the dynamic. From a Ground Truth History
    """
    if rids is None:
        rids = hist_gt[-1].robotsx.keys()
    hd = {k: np.asarray([h.robotsx[k][0] for h in hist_gt]) for k in rids}
    return hd

def _get_lm_ids(hist) -> list:
    ks = list(hist[-1].landmarks.keys())
    return ks

def get_lm_xye(hist, lm_id : int) -> np.ndarray:
    lm_idx = _get_lm_idx(hist, lm_id)
    xye = hist[-1].xest[lm_idx : lm_idx + 2]
    return xye

def plot_gt(hist, *args, block=None, **kwargs):
    """
        !
    """
    xyt = _get_xyt_true(hist)
    plt.plot(xyt[:, 0], xyt[:, 1], *args, **kwargs)
    if block is not None:
        plt.show(block=block)

def plot_dyn_gt(hist_gt, *args, block=None, rids : list = None, **kwargs):
    """
        ! 
    """
    hd = _get_dyn_lm_xyt(hist_gt, rids)
    for k, v in hd.items():
        kwargs["label"] = "rob: {}".format(k)
        plt.plot(v[:,0], v[:,1], *args, **kwargs)
    if block is not None:
        plt.show(block=block)

def plot_t(hist_gt, *args, block=None, N : int=10) -> None:
    """
        Function to plot N markers with the time t on the plot - to show specific times and what happens with the updates then.
        :param hist_gt: history with markers where to plot the t 
        :param N: how many ts to plot
    """
    nhist = len(hist_gt)
    xyt = _get_xyt_true(hist_gt)
    for k in np.linspace(0, nhist - 1, N):
        k = round(k)
        x_loc = xyt[k]
        t = hist_gt[k].t
        plt.text(x_loc[0], x_loc[1], "{0:.3f}".format(t))

def _get_lm_idx(hist, lm_id : int) -> int:
    """
        :param hist: history from which to get the idx 
        :param lm_id: for which id to retrieve the idx
    
        :return: state_idx
        :rtype: int

        Function to get the state vector index of a landmark from a history
    """
    return hist[-1].landmarks[lm_id][2]

def _get_lm_midx(hist, lm_id : int) -> int:
    """
        Function to get the map vector index of a landmark from a history
    """
    lmidx = _get_lm_idx(hist, lm_id)
    return lmidx - 3

def _lm_in_hist(hist, lm_id : int) -> bool:
    """
        test if the landmark id exists in the history
    """
    return True if lm_id in hist[-1].landmarks else False

def get_dyn_idcs(cfg_d : dict, hist) -> int:
    """
        ! 
        Get the indices in the state vector of the landmarks that are considered dynamic BY THE ROBOT
    """
    dyn_lm_list = get_dyn_lms(cfg_d)
    if dyn_lm_list is not None:
        ks = [_get_lm_idx(hist, k) for k in dyn_lm_list if _lm_in_hist(hist, k)]
    # ks = list([v[2] for _, v in hist[-1].seen_robots.items()])
    else:
        ks = []
    return ks

def get_dyn_lms(cfg_d : dict) -> list:
    """
        !
        Get a list of the dynamic landmarks from an ekf dictionary.
        If the key does not exist, returns None - can be tested for existence
    """
    return cfg_d.get("dynamic_lms")

def get_dyn_idcs_map(cfg_d : dict, hist) -> int:
    """
        Get the indices of the robot that are considered dynamic
    """
    ks = get_dyn_idcs(cfg_d, hist)
    nk = [n - 3 for n in ks]
    return nk

# getting the start time for an index or an id
def get_id_start_t(hist, lm_id : int) -> int:
    if not isdatmo(hist):
        idx = _get_lm_idx(hist, lm_id)
        st = get_idx_start_t(hist, idx)
    else:
        st = get_datmo_start_t(hist, lm_id)
    return st

def get_idx_start_t(hist, idx : int) -> int:
    start_t = None
    for i, h in enumerate(hist):
        if len(h.xest) > idx:
            start_t = i
            break
    return start_t

def get_datmo_start_t(hist, did : int):
    """
        Function to get the start time of a datmo object
    """
    t = None
    for i, h in enumerate(hist):
        t = i
        if h.trackers is not None and did in h.trackers:
            break
    return t

def get_lm_vis(hist, lm_id : int) -> list:
    """
        Function to get the t when a landmark was observed
        
        :param lm_id: id of landmark to check
    
        :return: boolean list
        :rtype: list
    """
    v = [True if lm_id in h.landmarks else False for h in hist]
    return v       

def _get_dyn_xyt_est(hist, start_t : int, r_id : int) -> list:
    """
        Function to get the xyt of a dynamic index.
        Differentiates between DATMO and normal hist.
        DATMO already aligns to global frame 
    """
    if not isdatmo(hist):
        ind = _get_lm_idx(hist, r_id)
        l = np.asarray([h.xest[ind : ind +2] for h in hist[start_t:]])
    else:
        xyt_r = [h.xest[:3] for h in hist[start_t:]]
        xyt_k = [h.trackers[r_id].xest[:2] for h in hist[start_t:]]
        l = np.asarray([forward(rob_xyt, lm_xy) for rob_xyt, lm_xy in zip(xyt_r, xyt_k)])       # inverse - point lm_xy is recorded in rob_xyt -> so has to be inverted with rob_xyt to be in global frame
    return l

def get_dyn_xyt_est(hist, r_id : int) -> list:
    """
        Function to get the xyt of a dynamic id. 
        Differentiates between DATMO and normal hist
    """
    r_start = get_id_start_t(hist, r_id)
    if r_start == None:
        print("Robot {} never found in map!".format(r_id))
        xyt = None
    xyt = _get_dyn_xyt_est(hist, r_start, r_id)
    return xyt

def _get_dyn_P_est(hist, start_t : int, r_id : int) -> list:
    ind = _get_lm_idx(r_id)
    P = [h.Pest[ind: ind+2, ind : ind+2] for h in hist[start_t:]]
    return P

def get_dyn_P_est(hist, r_id : int) -> list:
    r_start = get_id_start_t(hist, r_id)
    if r_start == None:
        print("Robot {} never found in map!".format(r_id))
    P = _get_dyn_P_est(hist, r_start, r_id)
    return P

def _get_fp_idcs(hist, fp_list):
    # TODO - this doesn't seem right...
    ks = list([v[2] for k, v in hist[-1].landmarks.items() if k in fp_list])
    return ks

def get_fp_idcs_map(hist, fp_list) -> int:
    ks = _get_fp_idcs(hist, fp_list)
    nk = [n -3 for n in ks]
    return nk

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

# Section on Plotting the map
def _plot_map_est(x, P, marker=None, ellipse=None, confidence=0.95, block=None):
    plt.scatter(x[:, 0], x[:, 1], **marker)
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

def plot_map_est(hist, cfg : dict, tf : np.ndarray = None, marker=None, ellipse=None, confidence=0.95, block=None):
    """
        Function to plot the map estimates positions of static landmarks. If dynamic_map_idcs is None, will plot the last estimate of all map markers
        Will exclude all dynamic_map_idcs from the plotting.
    """
    dynamic_map_idcs = get_dyn_idcs_map(cfg, hist)
    
    # remove the robot from the history
    xest = hist[-1].xest[3:]
    Pest = hist[-1].Pest[3:,3:]

    if dynamic_map_idcs:
        mm = cfg.get("motion_model")
        state_length = mm.get("state_length")
        xest, Pest, _, _ = _split_states(xest, Pest, dynamic_map_idcs, state_length)
    xest = xest.reshape((-1,2))
    
    # apply transform 
    if tf is not None:
        xest = inverse(tf, xest.T).T

    _plot_map_est(xest, Pest, marker=marker, ellipse=ellipse, confidence=confidence, block=block)

# Section on plotting the xy of the robot itself
def plot_xy_est(hist, **kwargs):
    xyt = _get_r_xyt_est(hist)
    _plot_xy_est(xyt, **kwargs)

def _plot_xy_est(xyt, **kwargs):
    """
        Function to plot xy estimates of the robot.
    """
    plt.plot(xyt[:,0], xyt[:,1], **kwargs)

def plot_transformed_xy_est(hist, tf : np.ndarray, **kwargs):
    """
        Function to plot an estimated transform 
    """
    xyt = _get_r_xyt_est(hist)
    # xy_t = _apply_transform(xyt.T, R_e, t_e)
    xy_t = inverse(tf, xyt.T)
    _plot_xy_est(xy_t.T, **kwargs)

# Section on plotting the estimated dynamic landmark over time.
def has_dynamic_lms(cfg : dict) -> bool:
    """
        Function to test whether a filter has a dynamic landmark
    """
    return "dynamic_lms" in cfg

def plot_dyn_est(hist, cfg_d : dict, dyn_id = None, tf : np.ndarray = None, **kwargs):
    """
        Wrapper function to differentiate between DATMO plotting and plotting inside an EKF
    """
    if dyn_id is None:
        dids = get_dyn_lms(cfg_d)
    else:
        dids = [dyn_id]

    # for each dynamic landmark
    for did in dids:
        xyt = get_dyn_xyt_est(hist, did)
        kwargs["label"] = "{} est".format(did) 
        if tf is not None:
            xyt_t = inverse(tf, xyt.T)
            xyt = xyt_t.T
        kwargs["label"] = "rob: {} est {}".format(did, "tf" if tf is not None else "")
        _plot_xy_est(xyt, **kwargs)

def plot_ellipse(hist, rob_id : int = None, confidence=0.95, N=10, block=None, **kwargs):
    if rob_id is None:
        # Get the xyt of the robot
        xyt = _get_r_xyt_est(hist)
        P = _get_robot_P(hist)
    else:
        # get the robot index in the map
        xyt = get_dyn_xyt_est(hist, rob_id)
        P = get_dyn_P_est(hist, rob_id)
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



### newly inserted stuff here    
### section on static methods for calculating the offset - to get the ATE
def get_ignore_idcs(cfg_d : dict, simdict : dict) -> list:
    """
        Function to get the list of indices which to ignore when calculating the transformation between maps.
        Uses:
            dynamic idcs
            false positive indices
            ignored indices
    """
    dyns = [int(k) for k in simdict['dynamic'].keys()]
    ign = cfg_d['ignore_ids']
    dyn_m = cfg_d.get("dynamic_lms")
    ds = dyns + ign + dyn_m if dyn_m is not None else dyns + ign 
    return list(set(ds))

def get_transformation_arun(pt : np.ndarray, pe : np.ndarray) -> tuple:
    """
        Method to estimate the transformation between two known pointclouds using Horn's Method. Needs a correspondence between at least 3 points (in 3D)
        :params: pt: true point locations
                pe: estimated point locations
        careful! pt and pe are supposed to be column vectors, e.g. 2 x N matrices!
        use with x_est column vectors in same format:
        x_fit = R_e @ x_est + t_e
        Paper reference: https://ieeexplore.ieee.org/document/4767965
        Code reference 1 [here](https://github.com/gnastacast/dvrk_vision/blob/1b9f9b43f52291452e8f0c174ec64b000ab412da/src/dvrk_vision/rigid_transform_3d.py#L9)
        Code refence 2 [here](https://gist.github.com/scturtle/c3037529098338eccc403a9842870273)
        Code reference 3 [here](https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py)
    
    """
    # assert pt.shape == pe.shape, "Not the same shape."
    # assert pt.shape[0] == 2, "1 point per column vector!"
        
    # 1. Compute centroids
    pt_centre = pt.mean(axis=1)
    pe_centre = pe.mean(axis=1)

    # 2. Shift to centroids
    qe = pe - pe_centre[:,np.newaxis]
    qt = pt - pt_centre[:,np.newaxis]

    # 3. SVD - eqn. (11) arun
    H = qt @ qe.transpose()                 
    # H2 = qe @ qt.transpose()
    U, _ , Vt = np.linalg.svd(H)

    # 4. Get parameters out
    # R_e = U @ V
    R_e = Vt.T @ U.T
    if np.linalg.det(R_e) < 0:
        Vt[:, 1] *= -1
        R_e = Vt.T @ U.T
    # eqn. (10) arun
    t_e = pe_centre - (R_e @ pt_centre) 
    
    return  t_e, R_e

def calculate_ATE_arun(x_true : np.ndarray, x_est : np.ndarray, tf : np.ndarray) -> np.ndarray:
    """
        Function to calculate the ATE with parameters calculated via LS fitting.
        x_true and x_est are N x 3 ndarrays.
        Get converted to 2 x N ndarrays.
        tf is a transform of format [x, y, theta]
    """
    x_t = x_true[:,:2].transpose()
    x_e = x_est[:,:2].transpose()
    x_fit = inverse(tf, x_e) # inverse, tf is the transform from true to est, so has to be used inverted.
    x_ls = (x_t - x_fit) ** 2
    return x_ls 

def get_transform(hist, map_lms : LandmarkMap, ignore_idcs : list = []) -> np.ndarray:
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
        t_e, R_e = get_transformation_arun(p, q)           
        tf = tf_from_tR(t_e=t_e, R_e=R_e)
        return tf

def calculate_metrics(simdict : dict, ekf_hists : dict, gt_hist) -> dict:
    # Calculate ATE
    workspace = np.array(simdict['map']['workspace'])
    mp = np.array(simdict['map']['landmarks'])
    lm_map = LandmarkMap(map=mp, workspace=workspace)
    x_true = _get_xyt_true(gt_hist)

    ate_d = {
        "dynamic" : simdict["dynamic_count"],
        "static" : lm_map._nlandmarks,
        "time" : simdict["time"],             
        "seed" : simdict["seed"],
        "fp_count" : simdict["fp_count"],                             
    }
    for ekf_id, ekf_hist in ekf_hists.items():
        cfg_ekf = simdict[ekf_id]
        ign_idcs = get_ignore_idcs(cfg_ekf, simdict)
        ate, tf =  get_ATE(
            hist = ekf_hist,
            map_lms = lm_map,
            x_t = x_true,
            ignore_idcs = ign_idcs 
            )
        
        ate_d[ekf_id + "-ate"] = np.sqrt(ate.mean())
        if has_dynamic_lms(cfg_ekf) and "FP" not in ekf_id:
            ate_dyn = get_dyn_ATE(ekf_hist, gt_hist, cfg_ekf, tf)
            ate_dynamic = ate_dyn.sum()       # alternative - use the mean of them. But that's kind of a wrong way to look at it. should get accumulated error!
            ate_d[ekf_id + '-dyn_ATE'] = ate_dynamic

        # Safety distance SDE
        # ! if the filter estimates dynamic lms - not necessarily the same as having them... INC p.ex. also estimates them.
        dynids = estimates_dynamic_lms(gt_hist, ekf_hist, cfg_ekf)
        if dynids:
            safety_d = get_safety_distance(ekf_hist, gt_hist, cfg_ekf, list(dynids))
            ate_d[ekf_id + "-SDE"] = safety_d.sum()

        # get transformation parameters
        t_d, theta_d  = get_transform_offsets(tf, angle=True)
        if theta_d == 0: theta_d = np.nan   # safeguard for case where FP only has 1 static landmark
        ate_d[ekf_id + "-translation_dist"] = t_d
        ate_d[ekf_id + "-rotation_dist"] = theta_d

        # get overall uncertainty parameter P-norm
        # ate_d[ekf_id + "-detP"] = get_Pnorm(history=ekf_hist, k=-1)
    return ate_d

def estimates_dynamic_lms(gt_hist, ekf_hist, cfg_ekf) -> list | None:
    """
        to determine whether a filter estimates dynamic landmarks
    """
    
    true_dids = true_dynamic_ids(gt_hist)
    slm_ids = get_static_landmark_ids(ekf_hist)
    if has_dynamic_lms(cfg_ekf): slm_ids += cfg_ekf.get("dynamic_lms")
    inters = set(slm_ids).intersection(set(true_dids))
    return inters

def true_dynamic_ids(gt_hist) -> list:
    """
        Function to get the list of true dynamic ids
    """
    return list(gt_hist[-1].robotsx.keys())

def get_static_landmark_ids(ekf_hist) -> list:
    """
        Function to get the list of estimated landmark ids
    """
    return list(ekf_hist[-1].landmarks.keys())

def __plot_map_helper(p, q):
    ax = plt.figure().add_subplot()
    ax.scatter(p[0, :], p[1,:], label="truth", c="k", marker="x")
    ax.scatter(q[0, :], q[1,:], label="measurements", color="g")
    # ! wrong format -> 
    tf = get_transformation_arun(p, q)                # tf in format [x,y,theta] 
    # q_transf = _apply_transform(q, R_e, t_e)
    q_transf = forward(tf, q)
    ax.scatter(q_transf[0, :], q_transf[1,:], label="transformed", color="r")
    ax.legend()
    plt.savefig("map_test.png")

def get_ATE(hist, map_lms : LandmarkMap, x_t : np.ndarray, t : slice = None, ignore_idcs : list = []) -> tuple[np.ndarray, np.ndarray]:
    """
        Function to get the absolute trajectory error
        uses the staticmethod calculate_ATE
        if t is given, uses slice of t
    """
    x_e = _get_r_xyt_est(hist)

    if t is not None:
        x_t = x_t[:,t]
        x_e = x_e[:,t]

    # getting the transform parameters
    tf = get_transform(hist, map_lms, ignore_idcs)
    
    # get ate
    ate = calculate_ATE_arun(x_t, np.asarray(x_e), tf)
    return ate, tf

def get_dyn_ATE(hist, gt_hist, cfg_d : dict, tf : np.ndarray, ident : int = None) -> np.ndarray:
    """
        params:
        history object to get it from
        gt_hist to compare with
        transform to apply to the _estimate_ -> necesary, otherwise 
        optional:
        identity for which to get te trajectory estimate
    """

    if ident is None:
        # get ALL dynamic indices from the history obejct ->
        ident = get_dyn_lms(cfg_d)
    else:
        ident = [ident]
    
    # has to be done as a list because of different starting times
    ate_l = []
    x_t_dict = _get_dyn_lm_xyt(gt_hist)
    for i, did in enumerate(ident):
        x_t = x_t_dict[did]
        x_e = get_dyn_xyt_est(hist, r_id=did)
        
        # if id_start is larger than 0
        id_start = get_id_start_t(hist, did)
        if id_start:
            x_t = x_t[id_start:]
        
        ate_dyn = calculate_ATE_arun(x_t, x_e, tf)
        ate_dyn_red = np.sqrt(ate_dyn.mean())
        ate_l.append(ate_dyn_red)
    
    return np.asarray(ate_l)

def get_safety_distance(hist, gt_hist, cfg_d : dict, ident : int | list = None) -> np.ndarray:
    """
        Function to get the safety distance of a dynamic landmark
    """
    if ident is None:
        ident = get_dyn_lms(cfg_d)
    elif isinstance(ident, int):
        ident = [ident]
    
    safety_l = []
    x_t_dict = _get_dyn_lm_xyt(gt_hist)
    x_re = _get_r_xyt_est(hist)
    x_rt = _get_xyt_true(gt_hist)[:,:2]
    for i, did in enumerate(ident):
        x_t = x_t_dict[did][:,:2]
        x_e = get_dyn_xyt_est(hist, r_id=did)
        
        id_start = get_id_start_t(hist, did)
        id_start = 0 if id_start is None else id_start
        x_t = x_t[id_start:]
        _x_re = x_re[id_start:]
        _x_rt = x_rt[id_start:]
        
        # ! Calculate the metric
        sd = _calculate_distance_error(_x_rt, _x_re, x_t, x_e)
        safety_l.append(np.sqrt(sd.mean()))

    return np.asarray(safety_l)

def _calculate_distance_error(x_rt : np.ndarray, x_re : np.ndarray, x_t : np.ndarray, x_e : np.ndarray) -> np.ndarray:
    """
        Function to calculate the distance error as the sum of squared difference between the estimated and the true distance to the object.
    """
    d_t = vect_dist(x_rt, x_t)
    d_e = vect_dist(x_re, x_e)
    return (d_e - d_t)**2


def __get_offset(x_true : np.ndarray, x_est : np.ndarray) -> np.ndarray:
        """
            # ! unused
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
def get_t(history):
        """
        Get time from history

        :return: simulation time vector
        :rtype: ndarray(n)

        Return simulation time vector, starts at zero.  The timestep is an
        attribute of the ``robot`` object.

        :seealso: :meth:`run` :meth:`history`
        """
        return np.array([h.t for h in history])

def get_Pnorm(history, k=None):
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
        return np.sqrt(np.linalg.det(history[k].Pest))
    else:
        p = [np.sqrt(np.linalg.det(h.Pest)) for h in history]
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