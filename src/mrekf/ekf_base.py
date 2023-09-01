import numpy as np
from roboticstoolbox.mobile import VehicleBase
from roboticstoolbox import RangeBearingSensor
from roboticstoolbox.mobile.landmarkmap import LandmarkMap
from collections import namedtuple
from spatialmath import base
from mrekf.ekf_math import *

EKFLOG =  namedtuple("EKFlog", "t xest Pest odo z innov K landmarks")   # todo remove odo here
MR_EKFLOG = namedtuple("MREKFLog", "t xtrue robotsx xest odo Pest innov S K z_lm z_r seen_robots landmarks")
GT_LOG = namedtuple("GroundTruthLog", "t xtrue odo z robotsx")

class BasicEKF(object):
    """
        Basic EKF - this one actually just does the mathematical steps.
            does not need (which are done by simulation)
                * sensor -> just the model W
            needs:
                * robot -> for fx - TODO - could theoretically turn this into functional programming?
            Abstract class. There are two derived classes.
            class 1 is for dynamic objects, which gets a list of landmark idcs to treat certain objects with the dynamic object model
                * which can also be used to include false positives
            class 2 is the standard version, which can optionally receive a list of landmark_idxs to ignore.
                * which can be used to ignore the other robots - baseline
                * which can be used - by not using it - to include the other robot as a false negative
            (class 3 will be implemented, which changes the V model)
                * needs the binary bayes filter
    """

    def __init__(self, x0 : np.ndarray = None, P0 : np.ndarray = None, robot : tuple[VehicleBase, np.ndarray] = None, W : np.ndarray = None, history : bool = False, joseph : bool = True) -> None:
        self.x0 = x0
        self.P0 = P0
        assert robot, "No robot model given, cannot compute fx()."
        assert W.shape == (2,2), "shape of W is not correct. please double check"
        
        # estimated states
        self._W_est = W
        self._robot = robot[0]
        self._V_est = robot[1]

        # history keeping
        self._keep_history = history
        if history:
            self._htuple = EKFLOG
            self._history = []

        # state estimates
        self._x_est = x0
        self._P_est = P0

        # landmark mgmt
        self.landmarks = {}

        # joseph update form
        self._joseph = joseph
    
    # properties
    @property
    def x_est(self) -> np.ndarray:
        return self._x_est
    
    @property
    def P_est(self) -> np.ndarray:
        return self.P_est
    
    @property
    def W_est(self) -> np.ndarray:
        return self._W_est
    
    @property
    def V_est(self) -> np.ndarray:
        return self._V_est
    
    @property
    def robot(self) -> VehicleBase:
        return self._robot
    
    @property
    def history(self) -> list:
        return self._history
    
    @property
    def joseph(self) -> bool:
        return self._joseph

    @property
    def state_length(self) -> int:
        """
            overwrite
        """
        return 3 + 2 * (len(self.landmarks))

    # landmark management
    @property
    def landmarks(self) -> dict:
        return self._landmarks
   
    def landmark_index(self, lm_id : int) -> int:
        try:
            jx = self.landmarks[lm_id][2]
            return jx
        except KeyError:
            raise ValueError("Unknown lm: {}".format(lm_id))
    
    def _landmark_add(self, lm_id : int) -> None:
        pos = self.get_state_length()
        self.landmarks[lm_id] = [len(self.landmarks), 1, pos]
    
    def _landmark_increment(self, lm_id : int) -> None:
        self.landmarks[lm_id][1] += 1

    def _isseenbefore(self, lm_id : int) -> bool:
        return lm_id in self.landmarks
    
    # step function -> that actually does the update
    def step(self, t, odo, zk : dict):
        """
            Function to take a step:
                * predict
                * update
                * insert new LMs
                * keep history
                * return
        """
        x_est = self.x_est
        P_est = self.P_est

        # predict
        x_pred, P_pred = self.predict(odo)

        # split readings into seen and unseen and ignore the ones that are supposed to be ignored.
        seen, unseen = self.process_readings(zk)

        # update
        x_est, P_est, innov, K = self.update(x_pred, P_pred, seen)

        # insert new things
        x_est, P_est = self.extend(x_est, P_est, unseen)

        # store values
        self._x_est = x_est
        self._P_est = P_est

        # logging
        self.store_history(t, x_est, P_est, z, innov, K, landmarks)

        # return values
        return x_est, P_est

    # associated functions -> these will need some form of overwriting
    def store_history(self, t : float, x_est : np.ndarray, P_est : np.ndarray, z : dict, innov : np.ndarray, K : np.ndarray, landmarks : dict) -> None:
        """
            overwrite
        """
        if self._keep_history:
            hist = self._htuple(
                t,
                x_est.copy() if x_est is not None else None,
                P_est.copy() if P_est is not None else None,
                z.copy() if z is not None else None,
                innov.copy() if innov is not None else None,
                K.copy() if K is not None else None,
                landmarks.copy() if landmarks is not None else None
            )
            self.history.append(hist)

    # Prediction step function
    def predict(self, odo, x_est : np.ndarray, P_est : np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        x_pred = self.predict_x(x_est, odo)
        
        V = self._get_V(x_est)
        Fv = self._get_Fv(x_est, odo)
        Fx = self._get_Fx(x_est, odo)
        P_pred = predict_P(P_est, V, Fx, Fv)
        return x_pred, P_pred

    def predict_x(self, x_est : np.ndarray, odo) -> np.ndarray:
        """
            overwrite -> this needs to incorporate motion model things
        """
        xv_est = x_est[:3]
        xm_est = x_est[3:]
        xv_pred = self.robot.f(xv_est, odo)
        xm_pred = xm_est
        x_pred = np.r_[xv_pred, xm_pred]
        return x_pred

    def _get_Fx(self, x_est : np.ndarray, odo) -> np.ndarray:
        """
            overwrite - but call super()._get_Fx() first and then just append the bottom part!
        """
        dim = len(x_est)
        Fx = np.zeros((dim, dim))
        xv_est = x_est[:3]
        v_Fx = self.robot.Fx(xv_est, odo)
        Fx[:3, :3] = v_Fx
        return Fx

    def _get_Fv(self, x_est : np.ndarray, odo) -> np.ndarray:
        """
            overwrite - but call super()._get_Fv() first and then just append the bottom part!
        """
        dim = len(x_est)
        Fv = np.zeros((dim, dim-1))
        xv_est = x_est[:3]
        v_Fv = self.robot.Fv(xv_est, odo)
        Fv[:3, :2] = v_Fv
        return Fv

    def _get_V(self, x_est : np.ndarray) -> np.ndarray:
        """
            overwrite - but call super()._get_V() first and then just append the bottom part!
            overwrite - in the overwritten version make the V a property of each LM -> already scaled.
                LM can be objects that have an id, a map index, a counter + the V
        """
        dim = len(x_est) - 1
        Vm = np.zeros((dim, dim))
        V_v = self.V        # todo - this is different than in the OG implementation -double check if results are equivalent
        Vm[:2, :2] = V_v
        return Vm

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
    def predict_static(self, odo) -> tuple[np.ndarray, np.ndarray]:
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
    def split_readings(self, zk : dict) -> tuple[dict, dict]:
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

    def update_static(self, x_pred : np.ndarray, P_pred : np.ndarray, seen_readings : dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            update function with only assumed static Landmarks.
            Using the same functions as the other part
        """
        innovation = self.get_innovation(x_pred, seen_readings)
        Hx = self.get_Hx(x_pred, seen_readings)
        Hw = self.get_Hw(x_pred, seen_readings)
        x_len = int((len(x_pred) - 3) / 2)
        W_est = self.get_W_est(x_len)

        S = calculate_S(P_pred, Hx, Hw, W_est)
        K = calculate_K(Hx, S, P_pred)
        
        x_est = update_state(x_pred, K, innovation)
        x_est[2] = base.wrap_mpi_pi(x_est[2])
        if self._joseph:
            P_est = update_covariance_joseph(P_pred, K, W_est, Hx)
        else:
            P_est = update_covariance_normal(P_pred, S, K)

        return x_est, P_est, innovation, K

    def get_g_funcs(self, x_est : np.ndarray, lms : dict, n : int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def extend_static(self, x_est : np.ndarray, P_est : np.ndarray, unseen_readings : dict) -> tuple[np.ndarray, np.ndarray]:
        """
            function to extend the map with only assumed static landmarks
        """
        n_new = len(unseen_readings) * 2
        W_est_full = self.get_W_est(len(unseen_readings))
        xf, Gz, Gx = self.get_g_funcs(x_est, unseen_readings, n_new)
        x_est, P_est = extend_map(
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
                self.landmarks
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
       
