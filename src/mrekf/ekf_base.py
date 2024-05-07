import numpy as np
from tqdm import tqdm
from roboticstoolbox.mobile import VehicleBase
from roboticstoolbox.mobile.landmarkmap import LandmarkMap
from collections import namedtuple
from copy import deepcopy
from spatialmath import base
from mrekf.sensor import SensorModel
from mrekf.ekf_math import *

"""
    TODO -> include logging module and .verbose factor
    include loglevel
    TODO - logging
        * remove odo from EKFlog - is part of GT_Log from simulation
        * find way to store which lms are considered dynamic - is a dynamic thing, should be stored in experiment_settings.
"""

EKFLOG =  namedtuple("EKFlog", "description t xest Pest odo z innov K landmarks")   
# MR_EKFLOG = namedtuple("MREKFLog", "t xest Pest odo z innov K landmarks")
DATMOLOG = namedtuple("DATMOlog", "description t xest Pest odo z innov K landmarks trackers")   
TRACKERLOG = namedtuple("TrackerLog", "description t x_tf P_tf x_p P_p xest Pest innov K")
GT_LOG = namedtuple("GroundTruthLog", "t xtrue odo z robotsx")

class BasicEKF(object):
    """
        Basic EKF - this one actually just does the mathematical steps.
            does not need (which are done by simulation)
                * sensor -> just the model W
                TODO - also need sensor h to evaluate innovation! And Hx, Hp etc.!
                TODO: also for the g functions
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

    def __init__(self, description : str, x0 : np.ndarray = None, P0 : np.ndarray = None, robot : tuple[VehicleBase, np.ndarray] = None, sensor : tuple[SensorModel, np.ndarray] = None, history : bool = False, joseph : bool = True,
                ignore_ids : list = []
                ) -> None:
        self.x0 = x0
        self.P0 = P0
        assert robot, "No robot model given, cannot compute fx()."
        assert sensor[1].shape == (2,2), "shape of W is not correct. please double check"
        # estimated states
        self._sensor = sensor[0]
        self._W_est = sensor[1]
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
        self._ignore_ids = ignore_ids # -> landmarks to ignore in the update
        self._landmarks = {}

        # joseph update form
        self._joseph = joseph

        # description to store which version this is
        self._description = description
    
    def __str__(self):
        s = f"{self.description} of type: {self.__class__.__name__} object: {len(self._x_est)} states"

        def indent(s, n=2):
            spaces = " " * n
            return s.replace("\n", "\n" + spaces)

        if self.ignore_ids:
            s += indent("\nignored ids: " + str(self.ignore_ids))
        if self.robot is not None:
            s += indent("\nrobot: " + str(self.robot))
        if self.V_est is not None:
            s += indent("\nV_est:  " + base.array2str(self.V_est))

        if self.sensor is not None:
            s += indent("\nsensor: " + str(self.sensor))
        if self.W_est is not None:
            s += indent("\nW_est:  " + base.array2str(self.W_est))
        return s

    def __repr__(self) -> str:
        return str(self)

    # properties
    @property
    def description(self) -> str:
        return self._description
    
    @description.setter
    def description(self, value):
        self._description = value

    @property
    def is_dynamic(self) -> bool:
        return False

    @property
    def x_est(self) -> np.ndarray:
        return self._x_est
    
    @property
    def P_est(self) -> np.ndarray:
        return self._P_est
    
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
    def sensor(self) -> SensorModel:
        return self._sensor
    
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
            esnure that this is still with self.landmarks - because that gets overwritten everywhere!
        """
        return 3 + 2 * (len(self.landmarks))

    # landmark management
    @property
    def landmarks(self) -> dict:
        return self._landmarks

    @property
    def ignore_ids(self) -> list:
        return self._ignore_ids

    def landmark_index(self, lm_id : int) -> int:
        """
            Index of landmark in STATE
        """
        try:
            jx = self.landmarks[lm_id][2]
            return jx
        except KeyError:
            raise ValueError("Unknown lm: {}".format(lm_id))
        
    def landmark_mindex(self, lm_id : int) -> int:
        """
            index of landmark in MAP
        """
        return self.landmark_index(lm_id) - 3 
    
    def _landmark_add(self, lm_id : int) -> None:
        pos = self.state_length
        self.landmarks[lm_id] = [len(self.landmarks), 1, pos]
    
    def _landmark_increment(self, lm_id : int) -> None:
        self.landmarks[lm_id][1] += 1

    def _isseenbefore(self, lm_id : int) -> bool:
        return lm_id in self.landmarks
    
    # step function -> that actually does the update
    def step(self, t, odo, zk : dict, true_states : dict):
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
        x_pred, P_pred = self.predict(odo, x_est, P_est)

        # split readings into seen and unseen and ignore the ones that are supposed to be ignored.
        seen, unseen = self.process_readings(zk)

        # update
        x_est, P_est, innov, K = self.update(x_pred, P_pred, seen)

        # insert new things
        x_est, P_est = self.extend(x_est, P_est, unseen, true_states)

        # store values
        self._x_est = x_est
        self._P_est = P_est

        # logging
        self.store_history(t, x_est, P_est, odo, zk, innov, K, self.landmarks)

        # return values
        return x_est, P_est

    # associated functions -> these will need some form of overwriting
    def store_history(self, t : float, x_est : np.ndarray, P_est : np.ndarray, odo, z : dict, innov : np.ndarray, K : np.ndarray, landmarks : dict) -> None:
        """
            
        """
        if self._keep_history:
            hist = self._htuple(
                self.description,
                t,
                x_est.copy() if x_est is not None else None,
                P_est.copy() if P_est is not None else None,
                odo.copy() if odo is not None else None,
                z.copy() if z is not None else None,
                innov.copy() if innov is not None else None,
                K.copy() if K is not None else None,
                landmarks.copy() if landmarks is not None else None
            )
            self.history.append(hist)

    # Prediction step function
    def predict(self, odo, x_est : np.ndarray, P_est : np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        x_pred = self.predict_x(x_est, odo)
        
        V = self._get_V()
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
            Jacobian of f wrt to the state x
        """
        dim = len(x_est)
        Fx = np.eye(dim)
        xv_est = x_est[:3]
        v_Fx = self.robot.Fx(xv_est, odo)
        Fx[:3, :3] = v_Fx
        return Fx

    def _get_Fv(self, x_est : np.ndarray, odo) -> np.ndarray:
        """
            Jacobian of f wrt to the noise v
        """
        xv_est = x_est[:3]
        cols = 2 * (1 + len(self.landmarks))
        rows = len(x_est)
        Fv = np.zeros((rows, cols))
        Fvv = self.robot.Fv(xv_est, odo)
        Fv[:3,:2] = Fvv
        return Fv

    def _get_V(self) -> np.ndarray:
        """
            Noise Matrix V
            overwrite - for binary bayes filter in the overwritten version make the V a property of each LM -> already scaled.
                LM can be objects that have an id, a map index, a counter + the V
        """
        dim = (1  + len(self.landmarks)) * 2
        # dim = len(x_est) - 1
        Vm = np.zeros((dim, dim))
        V_v = self.V_est        # todo - this is different than in the OG implementation -double check if results are equivalent
        Vm[:2, :2] = V_v
        return Vm

    # Processing the readings function
    def process_readings(self, zk : dict) -> tuple[dict, dict]:
        seen = {}
        unseen = {}
        for lm_id, z in zk.items():
            # skip if intended to ignore
            if lm_id in self.ignore_ids: continue
            # check if already seen or not
            if self._isseenbefore(lm_id):
                seen[lm_id] = z
            else:
                unseen[lm_id] = z
        return seen, unseen

    # Updating section
    def update(self, x_pred : np.ndarray, P_pred : np.ndarray, seen : dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        innovation = self.get_innovation(x_pred, seen)
        Hx = self.get_Hx(x_pred, seen)
        Hw = self.get_Hw(x_pred, seen)

        # x_len = int((len(x_pred) - 3) /2) # old -> have to use. because otherwise the length of dyn landmarks will be neglected
        # x_len = len(self.landmarks) # new?!
        W_est = self.get_W_est(len(seen))

        S = calculate_S(P_pred, Hx, Hw, W_est)
        K = calculate_K(Hx, S, P_pred)

        x_est = update_state(x_pred, K, innovation)
        x_est[2] = base.wrap_mpi_pi(x_est[2])                   # TODO: also do this for the dynamic landmarks?
        if self.joseph:
            P_est = update_covariance_joseph(P_pred, K, W_est, Hx)
        else:
            P_est = update_covariance_normal(P_pred, S, K)
        
        return x_est, P_est, innovation, K

    def get_innovation(self, x_pred : np.ndarray, seen_lms : dict) -> np.ndarray:
        """
            Overwrite - does not need to happen, because m_ind + 2 only takes x and y, which are all we need for the updates!
        """
        innov = []
        xv_pred = x_pred[:3]
        for lm_id, z in seen_lms.items():
            # get the predicted value of what it should be
            lm_ind = self.landmark_index(lm_id)
            xf = x_pred[lm_ind : lm_ind+2]

            # calculate the difference of what it should be
            z_pred = self.sensor.h(xv_pred, xf) 
            inn = [z[0] - z_pred[0], base.wrap_mpi_pi(z[1] - z_pred[1])]

            # append it to the innovation vector
            innov += inn
            self._landmark_increment(lm_id)
        return np.array(innov)

    def get_Hx(self, x_pred : np.ndarray, seen : dict) -> np.ndarray:
        """
            overwrite to consider extended state length
            new function -> using innovation as a variable-length list, not a vector of fixed size
        """
        # number of states to distribute to
        cols = x_pred.size
        # number of observations to distribute
        rows = 2 * len(seen)

        Hx = np.zeros((rows, cols))
        start_ind = 0
        xv_pred = x_pred[:3]
        for lm_id in seen:
            s_ind = self.landmark_index(lm_id)
            xf = x_pred[s_ind : s_ind + 2]

            # get sub-matrices
            Hxv = self.sensor.Hx(xv_pred, xf)
            Hp_k = self.sensor.Hp(xv_pred, xf)
            
            # insert into large matrix
            Hx[start_ind : start_ind + 2, :3] = Hxv
            Hx[start_ind : start_ind + 2, s_ind : s_ind + 2] = Hp_k
            start_ind += 2
        
        return Hx

    def get_Hw(self, x_pred : np.ndarray, seen : dict) -> np.ndarray:
        """
            No Overwriting
        """
        len_obs = 2 * len(seen)
        Hw = np.eye(len_obs)
        return Hw
    
    def get_W_est(self, no_obs : int) -> np.ndarray:
        """
            :param no_obs - number of observations -> 1 observation = [r, phi] -> half of innovation length
            potentially overwrite - if W is not extending to the 2 additional states. Check math again
        """
        _W = self._W_est
        W = np.kron(np.eye(no_obs), _W)
        return W

    # Section on Extending the map!
    def extend(self, x_est : np.ndarray, P_est : np.ndarray, unseen : dict, true_states : dict) -> tuple[np.ndarray, np.ndarray]:
        """
            overwrite - maybe -> depending if we find a better way to deal with the 2 in the state length and the W_est
            could set a state-length variable that is 2? and the 
        """
        W_est_full = self.get_W_est(len(unseen))
        xf, Gz, Gx = self.get_g_funcs(x_est, unseen, true_states)
        x_ext, P_ext = extend_map(
            x_est, P_est, xf, Gz, Gx, W_est_full
        )
        return x_ext, P_ext

    def get_g_funcs(self, x_est : np.ndarray, unseen : dict, true_states : dict = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Overwrite
            also: this needs sensor g functions
        """
        n = len(unseen) * 2
        Gx = np.zeros((n, 3))
        Gz = np.zeros((n, n))
        xf = np.zeros(n)
        xv = x_est[:3]
        for i, (lm_id, z) in enumerate(unseen.items()):
            xf_i = self.sensor.g(xv, z)
            Gz_i = self.sensor.Gz(xv, z)
            Gx_i = self.sensor.Gx(xv, z)

            xf[i * 2 : i* 2 + 2] = xf_i
            Gz[i * 2 : i * 2 + 2, i*2 : i*2+2] = Gz_i
            Gx[i*2 : i*2 + 2, :] = Gx_i
            
            #  map_index  is inside the functions. careful mgmt necessary!
            self._landmark_add(lm_id)
        
        return xf, Gz, Gx

    def _reset_filter(self):
        """
            Function to reset the filter, such that it can be re-instantiated from scratch 
        """
        self._history = []
        self._landmarks = {}
        self._x_est = deepcopy(self.x0)
        self._P_est = deepcopy(self.P0)
    
    def rerun_from_hist(self, gt_hist : list):
        """
            Function to rerun filter with same settings from a GroundTruh histoy.
        """
        print(f"Rerunning filter {self.description}")
        self._reset_filter()
        assert not self.history, "History object is not empty -> double check this worked"
        assert np.array_equal(self.x_est, self.x0), "x_est not x0 -> double check reset worked"
        assert np.array_equal(self.P_est, self.P0), "P_est not P0 -> double check reset worked"
        assert not self.landmarks, "Landmarks are still set -> check reset worked"
        dt = gt_hist[0].t
        [self.step(h.t - dt, h.odo, h.z, h.robotsx) for h in tqdm(gt_hist)]