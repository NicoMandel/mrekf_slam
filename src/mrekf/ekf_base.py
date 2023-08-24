from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from roboticstoolbox.mobile import VehicleBase
from roboticstoolbox import RangeBearingSensor
from roboticstoolbox.mobile.landmarkmap import LandmarkMap
from scipy.linalg import block_diag
from collections import namedtuple
from spatialmath import base

EKFLOG =  namedtuple("EKFlog", "t xest Pest odo z innov K")
MR_EKFLOG = namedtuple("MREKFLog", "t xtrue robotsx xest odo Pest innov S K z_lm z_r seen_robots")

### standard EKF algorithm that just does the prediction and the steps
class EKF_base(object):
    """
        basic EKF algorithm that just does the mathematical steps
        mathematical methods for the normal steps are exposed as static methods so that they can be reused by the MR_EKF
    """

    def __init__(self, x0 : np.ndarray = None, P0 : np.ndarray = None, robot : VehicleBase = None, sensor : RangeBearingSensor = None, history : bool = False, joseph : bool = True) -> None:
        self.x0 = x0
        self.P0 = P0
        # base models
        assert robot, "No robot model given, cannot compute prediction functions"
        assert sensor, "No sensor model given, cannot compute observation functions"
        self._robot = robot[0]
        self._sensor = sensor[0]

        self._V_est = robot[1]
        self._W_est = sensor[1] 

        self._keep_history = history  #  keep history
        if history:
            self._htuple = EKFLOG 
            self._history = []
        
        # initial estimate variables
        self._x_est = x0
        self._P_est = P0

        # landmark mgmt
        self._landmarks = {}

        # joseph update form
        self._joseph = joseph
    
    # properties
    @property
    def x_est(self):
        return self._x_est

    @property
    def P_est(self):
        return self._P_est

    @property
    def W_est(self):
        return self._W_est

    @property
    def V_est(self):
        return self._V_est

    @property
    def sensor(self):
        return self._sensor
    
    @property
    def robot(self):
        return self._robot
    
    @property
    def history(self):
        return self._history
    
    @property
    def joseph(self):
        return self._joseph

   
    # landmark housekeeping
    def get_state_length(self):#
        return 3 + 2 * len(self._landmarks)

    @property
    def landmarks(self):
        return self._landmarks
    
    def landmark_index(self, lm_id : int) -> int:
        try:
            jx = self._landmarks[lm_id][2]
            return jx
        except KeyError:
            raise ValueError("Unknown lm: {}".format(lm_id))

    def _landmark_add(self, lm_id):
        pos = self.get_state_length()
        self.landmarks[lm_id] = [len(self._landmarks), 1, pos]
    
    def _landmark_increment(self, lm_id):
        self._landmarks[lm_id][1] += 1  # update the count
    
    def _isseenbefore(self, lm_id):
        return lm_id in self._landmarks
    
    def landmark_x(self, id):
        """
        straight from PC
        Landmark position

        :param id: landmark index
        :type id: int
        :return: landmark position :math:`(x,y)`
        :rtype: ndarray(2)

        Returns the landmark position from the current state vector.
        """
        jx = self.landmark_index(id)
        return self._x_est[jx : jx + 2]

    # functions working with the models
    # prediction function
    def predict_static(self, odo) -> Tuple[np.ndarray, np.ndarray]:
        """
            basic version of predicting, assuming only dynamic primary robot and static LMs
        """
        xv_est = self.x_est[:3]
        xm_est = self.x_est[3:]
        Pvv_est = self.P_est[:3, :3]
        Pmm_est = self.P_est[3:, 3:]
        # covariance
        Pvm_est = self.P_est[:3, 3:]

        # transition functions
        xv_pred = self.robot.f(xv_est, odo)
        Fx = self.robot.Fx(xv_est, odo)
        Fv = self.robot.Fv(xv_est, odo)

        # predict Vehicle
        Pvv_pred = Fx @ Pvv_est @ Fx.T + Fv @ self.V_est @ Fv.T
        Pvm_pred = Fx @ Pvm_est

        # map parts stay the same
        Pmm_pred = Pmm_est
        xm_pred = xm_est

        x_pred = np.r_[xv_pred, xm_pred]
        P_pred = np.block([
            [Pvv_pred,  Pvm_pred],
            [Pvm_pred.T, Pmm_pred]
        ])

        return x_pred, P_pred

    # split function
    def split_readings(self, zk : dict) -> Tuple[dict, dict]:
        seen = {}
        unseen = {}
        for lm_id, z in zk.items():
            if self._isseenbefore(lm_id):
                seen[lm_id] = z
            else:
                unseen[lm_id] = z
        
        return seen, unseen

    # updating functions
    def get_innovation(self, x_pred : np.ndarray, seen_rds : dict) -> np.ndarray:
        """
            returns the innovation of all seen readings.
            also includes the increment of the landmark
        """
        innov = np.zeros(len(x_pred) - 3)
        xv_pred = x_pred[:3]
        for lm_id, z in seen_rds.items():
            m_ind = self.landmark_index(lm_id)
            xf = x_pred[m_ind : m_ind + 2]
            z_pred = self.sensor.h(xv_pred, xf)
            inn = inn = np.array(
                    [z[0] - z_pred[0], base.wrap_mpi_pi(z[1] - z_pred[1])]
                )
            # m_ind is the index in the full state vector
            innov[m_ind - 3 : m_ind-1] = inn
            self._landmark_increment(lm_id)
        
        return innov

    def get_Hx(self, x_pred : np.array, seen_readings : dict) -> np.ndarray:
        dim = x_pred.size
        Hx = np.zeros((dim-3, dim))
        xv_pred = x_pred[:3]

        for lm_id, _ in seen_readings.items():
            l_ind = self.landmark_index(lm_id)
            xf = x_pred[l_ind : l_ind+2]

            Hp_k = self.sensor.Hp(xv_pred, xf)
            Hxv = self.sensor.Hx(xv_pred, xf)

            l_mind = l_ind -3
            Hx[l_mind : l_mind+2, :3] = Hxv
            Hx[l_mind : l_mind+2, l_ind : l_ind + 2] = Hp_k
        
        return Hx

    def get_Hw(self, x_pred : np.array, seen_readings : dict) -> np.ndarray:
        """
            Only object measurements
        """
        Hw = np.eye(x_pred.size -3)
        return Hw

    def get_W_est(self, x_len : int) -> np.ndarray:
        _W = self._W_est
        W = np.kron(np.eye(int(x_len), dtype=int), _W)
        return W

    def update_static(self, x_pred : np.ndarray, P_pred : np.ndarray, seen_readings : dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            update function with only assumed static Landmarks.
            Using the same functions as the other part
        """
        innovation = self.get_innovation(x_pred, seen_readings)
        Hx = self.get_Hx(x_pred, seen_readings)
        Hw = self.get_Hw(x_pred, seen_readings)
        x_len = int((len(x_pred) - 3) / 2)
        W_est = self.get_W_est(x_len)

        S = EKF_base.calculate_S(P_pred, Hx, Hw, W_est)
        K = EKF_base.calculate_K(Hx, S, P_pred)
        
        x_est = EKF_base.update_state(x_pred, K, innovation)
        x_est[2] = base.wrap_mpi_pi(x_est[2])
        if self._joseph:
            P_est = EKF_base.update_covariance_joseph(P_pred, K, W_est, Hx)
        else:
            P_est = EKF_base.update_covariance_normal(P_pred, S, K)

        return x_est, P_est, innovation, K

    def get_g_funcs(self, x_est : np.ndarray, lms : dict, n : int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Gx = np.zeros((n,3))
        Gz = np.zeros((n, n))
        xf = np.zeros(n)
        xv = x_est[:3]
        for i, (lm_id, z) in enumerate(lms.items()):
            xf_i = self.sensor.g(xv, z)
            Gz_i = self.sensor.Gz(xv, z)
            Gx_i = self.sensor.Gx(xv, z)

            xf[i * 2 : i*2 + 2] = xf_i
            Gz[i*2 : i*2 + 2, i*2 : i*2 + 2] = Gz_i
            Gx[i*2 : i*2 + 2, :] = Gx_i

            self._landmark_add(lm_id)
        
        return xf, Gz, Gx

    def extend_static(self, x_est : np.ndarray, P_est : np.ndarray, unseen_readings : dict) -> Tuple[np.ndarray, np.ndarray]:
        """
            function to extend the map with only assumed static landmarks
        """
        n_new = len(unseen_readings) * 2
        W_est_full = self.get_W_est(len(unseen_readings))
        xf, Gz, Gx = self.get_g_funcs(x_est, unseen_readings, n_new)
        x_est, P_est = EKF_base.extend_map(
            x_est, P_est, xf, Gz, Gx, W_est_full
        )
        return x_est, P_est
    
    def step(self, t, odo, zk : dict):
        """
            Function to take a step:
                * predict
                * update
                * insert new LMs
                * return
        """
        x_est = self.x_est
        P_est = self.P_est

        # predict
        x_pred, P_pred = self.predict_static(odo)

        # split readings into seen and unseen
        seen, unseen = self.split_readings(zk)

        # update
        x_est, P_est, innov, K = self.update_static(x_pred, P_pred, seen)

        # insert new things
        x_est, P_est = self.extend_static(x_est, P_est, unseen)

        # store values
        self._x_est = x_est
        self._P_est = P_est

        # logging
        if self._keep_history:
            hist = self._htuple(
                t,
                x_est.copy(),
                P_est.copy(),
                odo.copy(),
                zk.copy() if zk is not None else None,
                innov.copy() if innov is not None else None,
                K.copy() if K is not None else None,
            )
            self._history.append(hist)

        # return values
        return x_est, P_est


    # Plotting section
    def get_xyt(self):
        r"""
        Get estimated vehicle trajectory

        :return: vehicle trajectory where each row is configuration :math:`(x, y, \theta)`
        :rtype: ndarray(n,3)

        :seealso: :meth:`plot_xy` :meth:`run` :meth:`history`
        """
        xyt = np.array([h.xest[:3] for h in self._history])
        return xyt

    def plot_xy(self, *args, block=None, **kwargs):
        """
        Plot estimated vehicle position

        :param args: position arguments passed to :meth:`~matplotlib.axes.Axes.plot`
        :param kwargs: keywords arguments passed to :meth:`~matplotlib.axes.Axes.plot`
        :param block: hold plot until figure is closed, defaults to None
        :type block: bool, optional

        Plot the estimated vehicle path in the xy-plane.

        :seealso: :meth:`get_xyt` :meth:`plot_error` :meth:`plot_ellipse` :meth:`plot_P`
            :meth:`run` :meth:`history`
        """
        xyt = self.get_xyt()
        plt.plot(xyt[:, 0], xyt[:, 1], *args, **kwargs)
        if block is not None:
            plt.show(block=block)
        
    def plot_map(self, marker=None, ellipse=None, confidence=0.95, block=None):
        """
        taken directly from PC
        Plot estimated landmarks

        :param marker: plot marker for landmark, arguments passed to :meth:`~matplotlib.axes.Axes.plot`, defaults to "r+"
        :type marker: dict, optional
        :param ellipse: arguments passed to :meth:`~spatialmath.base.graphics.plot_ellipse`, defaults to None
        :type ellipse: dict, optional
        :param confidence: ellipse confidence interval, defaults to 0.95
        :type confidence: float, optional
        :param block: hold plot until figure is closed, defaults to None
        :type block: bool, optional

        Plot a marker  and covariance ellipses for each estimated landmark.

        :seealso: :meth:`get_map` :meth:`run` :meth:`history`
        """
        if marker is None:
            marker = {
                "marker": "+",
                "markersize": 10,
                "markerfacecolor": "red",
                "linewidth": 0,
            }

        xm = self._x_est[3:]
        P = self._P_est[3:,3:]

        # mark the estimate as a point
        xm = xm.reshape((-1, 2))  # arrange as Nx2
        plt.plot(xm[:, 0], xm[:, 1], **marker)

        # add an ellipse
        if ellipse is not None:
            for i in range(xm.shape[0]):
                Pi = self.P_est[i : i + 2, i : i + 2]
                # put ellipse in the legend only once
                if i == 0:
                    base.plot_ellipse(
                        Pi,
                        centre=xm[i, :],
                        confidence=confidence,
                        inverted=True,
                        label=f"{confidence*100:.3g}% confidence",
                        **ellipse,
                    )
                else:
                    base.plot_ellipse(
                        Pi,
                        centre=xm[i, :],
                        confidence=confidence,
                        inverted=True,
                        **ellipse,
                    )
        # plot_ellipse( P * chi2inv_rtb(opt.confidence, 2), xf, args{:});
        if block is not None:
            plt.show(block=block)

    def plot_ellipse(self, confidence=0.95, N=10, block=None, **kwargs):
        """
        Straight from PC
        Plot uncertainty ellipses

        :param confidence: ellipse confidence interval, defaults to 0.95
        :type confidence: float, optional
        :param N: number of ellipses to plot, defaults to 10
        :type N: int, optional
        :param block: hold plot until figure is closed, defaults to None
        :type block: bool, optional
        :param kwargs: arguments passed to :meth:`spatialmath.base.graphics.plot_ellipse`

        Plot ``N`` uncertainty ellipses spaced evenly along the trajectory.

        :seealso: :meth:`get_P` :meth:`run` :meth:`history`
        """
        nhist = len(self._history)

        if "label" in kwargs:
            label = kwargs["label"]
            del kwargs["label"]
        else:
            label = f"{confidence*100:.3g}% confidence"

        for k in np.linspace(0, nhist - 1, N):
            k = round(k)
            h = self._history[k]
            if k == 0:
                base.plot_ellipse(
                    h.Pest[:2, :2],
                    centre=h.xest[:2],
                    confidence=confidence,
                    label=label,
                    inverted=True,
                    **kwargs,
                )
            else:
                base.plot_ellipse(
                    h.Pest[:2, :2],
                    centre=h.xest[:2],
                    confidence=confidence,
                    inverted=True,
                    **kwargs,
                )
        if block is not None:
            plt.show(block=block)
    
    def disp_P(self, t :int = -1, colorbar=False):
        """
        Display covariance matrix

        :param t: timestep
        :type P: ndarray(n,n)
        :param colorbar: add a colorbar
        :type: bool or dict

        Plot the elements of the covariance matrix as an image. If ``colorbar``
        is True add a color bar, if `colorbar` is a dict add a color bar with
        these options passed to colorbar.

        .. note:: A log scale is used.

        :seealso: :meth:`~matplotlib.axes.Axes.imshow` :func:`matplotlib.pyplot.colorbar`
        """
        P_hist = [h.Pest for h in self.history]
        P = P_hist[t]
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

    ### Section on Evaluation
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
    
    def get_transform(self, map_lms : LandmarkMap) -> Tuple[np.array, np.ndarray, float]:
        """
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
            if lm_id > 99: continue     # robots are inserted with lms > 99
            p.append(map_lms[lm_id])
            q.append(self.landmark_x(lm_id))

        p = np.array(p).T
        q = np.array(q).T

        return self.get_transformation_params(q, p)
    
    def get_ATE(self, map_lms : LandmarkMap, t : slice = None) -> np.ndarray:
        """
            Function to get the absolute trajectory error
            uses the staticmethod calculate_ATE
            if t is given, uses slice of t
        """

        x_t = self.robot.x_hist
        x_e = self.get_xyt()

        if t is not None:
            x_t = x_t[:,t]
            x_e = x_e[:,t]

        # getting the transform parameters
        c, Q, s = self.get_transform(map_lms)

        return self.calculate_ATE(x_t, x_e, s, Q, c)

    ### section with static methods - pure mathematics, just gets used by every instance
    @staticmethod
    def predict(x_est : np.ndarray, P_est : np.ndarray, robot : VehicleBase, odo, Fx : np.ndarray, Fv : np.ndarray, V : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            prediction function just mathematically
            requires the correct Fx and Fv dimensions
            V is the noise variance matrix. Is constructed by inserting the correct V in the right places.
            which means that for every dynamic landmark, a V must be inserted at the diagonal block. 
        """
        # state prediction - only predict the vehicle in the non-kinematic case. 
        xv_est = x_est[:3]
        xm_est = x_est[3:]
        xm_pred = xm_est
        xv_pred = robot.f(xv_est, odo)
        x_pred = np.r_[xv_pred, xm_pred]
        P_est = P_est
        
        # Covariance prediction
        # differs slightly from PC. see
        # [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/EKF.py]]
        # L 806
        P_pred = Fx @ P_est @ Fx.T + Fv @ V @ Fv.T

        return x_pred, P_pred

    ### update steps
    @staticmethod
    def calculate_S(P_pred : np.ndarray, Hx : np.ndarray, Hw : np.ndarray, W_est : np.ndarray) -> np.ndarray:
        """
            function to calculate S
        """
        S = Hx @ P_pred @ Hx.T + Hw @ W_est @ Hw.T
        return S
    
    @staticmethod
    def calculate_K(Hx : np.ndarray, S : np.ndarray, P_pred : np.ndarray) -> np.ndarray:
        """
            Function to calculate K
        """
        K = P_pred @ Hx.T @ np.linalg.inv(S)
        return K
    
    @staticmethod
    def update_state(x_pred : np.ndarray, K : np.ndarray, innov : np.ndarray) -> np.ndarray:
        """
            Function to update the predicted state with the Kalman gain and the innovation
            K or innovation for not measured landmarks should both be 0
        """
        x_est = x_pred + K @ innov
        return x_est

    @staticmethod
    def update_covariance_joseph(P_pred : np.ndarray, K : np.ndarray, W_est : np.ndarray, Hx : np.ndarray) -> np.ndarray:
        """
            Function to update the covariance
        """
        I = np.eye(P_pred.shape[0])
        P_est = (I - K @ Hx) @ P_pred @ (I - K @ Hx).T + K @ W_est @ K.T
        return P_est
    
    @staticmethod
    def update_covariance_normal(P_pred : np.ndarray, S : np.ndarray, K : np.ndarray) -> np.ndarray:
        """
            update covariance
        """
        P_est = P_pred - K @ S @ K.T
        # symmetry enforcement
        P_est = 0.5 * (P_est + P_est.T)
        return P_est
    
    ### inserting new variables
    @staticmethod
    def extend_map(x : np.ndarray, P : np.ndarray, xf : np.ndarray, Gz : np.ndarray, Gx : np.ndarray, W_est : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # extend the state vector with the new features
        x_ext = np.r_[x, xf]

        # extend the map
        n_x = len(x)
        n_lm = len(xf)

        Yz = np.block([
            [np.eye(n_x), np.zeros((n_x, n_lm))    ],
            [Gx,        np.zeros((n_lm, n_x-3)), Gz]
        ])
        P_ext = Yz @ block_diag(P, W_est) @ Yz.T

        return x_ext, P_ext
    
    ### section on static methods for calculating the offset - to get the ATE
    @staticmethod
    def get_transformation_params(p1 : np.ndarray, p2 : np.ndarray) -> Tuple[np.array, np.ndarray, float]:
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
    
    @staticmethod
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

    @staticmethod
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
    
    @staticmethod
    def compare_update(h1 : namedtuple, h2 : namedtuple, t : slice = None) -> np.ndarray:
        """
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
