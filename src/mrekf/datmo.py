import numpy as np
from copy import deepcopy
from roboticstoolbox.mobile import VehicleBase, RangeBearingSensor
from mrekf.ekf_base import BasicEKF, DATMOLOG, TRACKERLOG
from mrekf.motionmodels import BaseModel, KinematicModel, BodyFrame
from mrekf.ekf_math import *
from spatialmath import base

class Tracker(object):
    """
        Tracker object to be used with DATMO.
        Each dynamic landmark receives its own Tracker object
    """

    def __init__(self, ident: int,  motion_model : BaseModel, sensor : RangeBearingSensor, dt : float, x0 : np.ndarray, P0 : np.ndarray, theta : float) -> None:
        self._theta_p = theta

        self._mm = motion_model
        self._sensor = sensor
        self._id = ident
        self._dt = dt
        self._x_est = x0
        self._P_est = P0
        self._K = None
        self._innovation = None

        # intermediate values
        self._x_tf = deepcopy(x0)
        self._x_p = deepcopy(x0)
        
        self._P_tf = deepcopy(P0)
        self._P_p = deepcopy(P0)

    def __str__(self):
        s = f"{self.id} id - {self.__class__.__name__} object: {len(self._x_est)} states"

        def indent(s, n=2):
            spaces = " " * n
            return s.replace("\n", "\n" + spaces)
        s+= indent(f"\ndt:{self.dt}")
        s += indent("\nMotion Model:  " + str(self.motion_model))
        if self.sensor is not None:
            s += indent("\nsensor: " + str(self.sensor))
        
        return s

    def __repr__(self) -> str:
        return str(self)

    @property
    def id(self) -> int:
        return self._id
    
    @property
    def motion_model(self) -> BaseModel:
        return self._mm
    
    @property
    def sensor(self) -> RangeBearingSensor:
        return self._sensor

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def x_est(self) -> np.ndarray:
        return self._x_est
    
    @property
    def P_est(self) -> np.ndarray:
        return self._P_est
    
    @property
    def x_tf(self) -> np.ndarray:
        return self._x_tf

    @property
    def x_p(self) -> np.ndarray:
        return self._x_p

    @property
    def P_tf(self) -> np.ndarray:
        return self._P_tf

    @property
    def P_p(self) -> np.ndarray:
        return self._P_p 
    
    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def innovation(self) -> np.ndarray:
        return self._innovation
    
    def is_kinematic(self) -> bool:
        k = True if isinstance(self.motion_model, KinematicModel) or isinstance(self.motion_model, BodyFrame) else False
        return k

    def transform(self, odo) -> tuple[np.ndarray, np.ndarray]:
        """
            transform requires an increase in uncertainty, see Sola p. 154
        """
        x_est = self.x_est
        P_est = self.P_est

        # transform state
        x_tf = self.motion_model.j(x_est, odo)
        
        # transform covariance
        Jo = self.motion_model.Jo(x_est, odo)
        Ju = self.motion_model.Ju(x_est, odo)
        V = self.motion_model.V
        P_tf = Jo @ P_est @ Jo.T + Ju @ V @ Ju.T

        # write states
        self._x_tf = x_tf
        self._P_tf = P_tf


        return x_tf, P_tf 

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        # get the transformed states
        x_e = self.x_tf
        P_e = self.P_tf
        
        # predict the state
        x_pred = self.motion_model.f(x_e)
        
        # predict the covariance
        V = self.motion_model.V
        Fx = self.motion_model.Fx(x_e)
        Fv = self.motion_model.Fv(x_e)
        P_pred = predict_P(P_e, V, Fx, Fv)
        
        # return the new state and covariance
        self._x_p = x_pred
        self._P_p = P_pred
        return x_pred, P_pred
    
    def update(self, obs) -> tuple[np.ndarray, np.ndarray]:
        x_pred = self._x_p
        P_pred = self._P_p

        x_p = x_pred[:2]

        # Since all update are performed in the local robot frame at time k, the xv_est is 0,0,0
        xv_est = np.array([0., 0., 0.,])
        # Get the innovation
        inn = self._get_innovation(xv_est, x_p, obs)
        
        # Get Hx
        Hx = self._get_Hx(xv_est, x_p)

        # get Hw
        Hw = self._get_Hw(obs)

        # Get W_est
        W_est = self._get_W_est()

        # perform the update 
        S = calculate_S(P_pred, Hx, Hw, W_est)
        K = calculate_K(Hx, S, P_pred)
        x_est = update_state(x_pred, K, inn)

        if isinstance(self.motion_model, BodyFrame):
            x_est[3] = base.wrap_mpi_pi(x_est[3])
        
        P_est = update_covariance_joseph(P_pred, K, W_est, Hx)
        
        # Storing values for the history 
        self._x_est = x_est
        self._P_est = P_est
        self._innovation = inn
        self._K = K
        return x_est, P_est

    def _get_innovation(self, xv_est : np.ndarray, x_e : np.ndarray, z : np.ndarray) -> np.ndarray:
        z_pred = self.sensor.h(xv_est, x_e)
        inn = [z[0] - z_pred[0], base.wrap_mpi_pi(z[1] - z_pred[1])]
        return inn
    
    def _get_Hx(self, xv_est : np.ndarray, x_e : np.ndarray) -> np.ndarray:
        is_kin = self.is_kinematic()
        Hp = self.sensor.Hp(xv_est, x_e, is_kin)
        return Hp

    def _get_Hw(self, z) -> np.ndarray:
        Hw = np.eye(len(z))
        return Hw

    def _get_W_est(self) -> np.ndarray:
        # ! this is wrong, will return null
        return self.sensor._W

class DATMO(BasicEKF):

    def __init__(self, description : str, 
                 dynamic_ids: list, motion_model : BaseModel,  x0: np.ndarray = np.array([0., 0., 0.]), P0: np.ndarray = None, robot: tuple[VehicleBase, np.ndarray] = None, sensor: tuple[RangeBearingSensor, np.ndarray] = None,
                history: bool = False, joseph: bool = True, ignore_ids: list = [], use_true : bool = False) -> None:
        super().__init__(description, x0, P0, robot, sensor, history, joseph, ignore_ids)
        self._dynamic_ids = dynamic_ids
        self._motion_model = motion_model

        self._dyn_objects = {}

        # adding dynamic objects
        self._use_true = use_true

        # overwriting the history parameter
        if history:
            self._htuple = DATMOLOG
            self._ttuple = TRACKERLOG

    def __str__(self):
        s = f"{self.description} of type {self.__class__.__name__} object: {len(self._x_est)} states"

        def indent(s, n=2):
            spaces = " " * n
            return s.replace("\n", "\n" + spaces)
        s+= indent("\n dynamic landmarks: " + str(self.dynamic_ids))
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

    def is_kinematic(self) -> bool:
        k = True if isinstance(self.motion_model, KinematicModel) or isinstance(self.motion_model, BodyFrame) else False
        return k

    @property
    def dyn_objects(self) -> dict[Tracker]:
        return self._dyn_objects

    @property
    def dynamic_ids(self) -> set:
        return self._dynamic_ids

    @property
    def seen_dyn_lms(self) -> list:
        return self.dyn_objects.keys()

    @property
    def seen_static_lms(self) -> set:
        return set(self.landmarks.keys()) - set(self.dynamic_ids)
    
    @property
    def use_true(self) -> bool:
        return self._use_true

    def dynamic_lms_in_dict(self, sd : dict) -> dict:
        return {k : v for k, v in sd.items() if k in self.dynamic_ids}

    def static_lms_in_dict(self, sd : dict) -> dict:
        return {k : v for k, v in sd.items() if k not in self.dynamic_ids}

    def split_observations(self, obs : dict) -> tuple[dict, dict]:
        """
            splitting observations into dynamic and static
        """
        dyn = self.dynamic_lms_in_dict(obs)
        stat = self.static_lms_in_dict(obs)
        return stat, dyn

    @property
    def motion_model(self) -> BaseModel:
        return self._motion_model
    
    def landmark_position(self, lm_id : int) -> int:
        """
            Position when the landmark was observed. Not state, but count of observed
            Needs addition of the vehicle state. e.g. first observed landmark has pos 0, but is in index 3, because behind Vehicle.
            Basically this ignores the dimension of dynamic landmarks, p.ex. when V has only 2 terms, but 4 states
        """
        try:
            jx = self.landmarks[lm_id][0]
            return jx
        except KeyError:
            raise ValueError("Unknown lm: {}".format(lm_id))

    def has_kinematic_model(self) -> bool:
        return True if self.motion_model.state_length > 2 else False

    # overwritten, to enable the use of lm_id
    def _isseenbefore(self, lm_id : int) -> bool:
        return lm_id in self.landmarks or lm_id in self.dyn_objects

    # Step has to be overwritten, because the transformation needs the odometry, part of the update step
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

        # update - changes made here!
        x_est, P_est, innov, K = self.update(x_pred, P_pred, seen, odo)

        # insert new things
        x_est, P_est = self.extend(x_est, P_est, unseen, true_states)

        # store values
        self._x_est = x_est
        self._P_est = P_est

        # logging
        self.store_history(t, x_est, P_est, odo, zk, innov, K, self.landmarks)

        # return values
        return x_est, P_est 

    def update(self, x_pred : np.ndarray, P_pred : np.ndarray, seen : dict, odo) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            Update function overwritten for DATMO. Updating the dynamic landmarks is done after the state update, with the new state.
        """
        # 1. split the observations into dynamic and static observations
        static, dynamic = self.split_observations(seen)
        
        # 2. perform the regular update with only the static landmarks
        x_est, P_est, innov, K = super().update(x_pred, P_pred, static)

        # 3. for each dynamic landmark - transform into new frame and update
        for ident, obs in dynamic.items():
            x_tf, P_tf = self.dyn_objects[ident].transform(odo)
            x_p, P_p = self.dyn_objects[ident].predict()
            x_e, P_e = self.dyn_objects[ident].update(obs)
            
        return x_est, P_est, innov, K

    def extend(self, x_est : np.ndarray, P_est : np.ndarray, unseen : dict, true_states : dict) -> tuple[np.ndarray, np.ndarray]:
        """
            extending the map by static landmarks
            adding new dynamic landmarks to each independent tracker
        """
        # 1. split the observations into dynamic and static observations
        static, dynamic = self.split_observations(unseen)
        
        # 2. insert the static observations
        x_est, P_est = super().extend(x_est, P_est, static, true_states=None)   # True states are not used for own ekf, only for each object init

        # 3. create a new tracker object for each dynamic landmark and add it to the dictionary
        for ident, obs in dynamic.items():
            x_n, P_n = self.init_dyn(obs, ident, true_states)

            theta = x_est[2]
            # 5. create a tracker and insert it
            nt = Tracker(ident, self.motion_model, self.sensor, self.robot.dt, x_n, P_n, theta)
            
            self.dyn_objects[ident] = nt


        return x_est, P_est
    
    def init_dyn(self, obs : np.ndarray, ident : int, true_states : dict) -> tuple[np.ndarray, np.ndarray]:
        """
            function to create a new object from a given observation
            Get from PC - inserting with known pose.
            However - difference - insertion not done in **global** frame, but in **local** robot frame,
            therefore redefining g and Gz.
        """
        kin = self.is_kinematic()
        init_val = None
        if kin:
            if self.use_true:
                x, v = true_states[ident]
                init_val = self.motion_model.get_true_state(v, base.wrap_mpi_pi(x[2]))
            else:
                init_val = self.motion_model.vmax
    
        # actual functions
        x_est = self._g(obs, kin, init_val)
        Gz = self._Gz(obs, kin)
        W_est = self.W_est
        P_est = Gz @ W_est @ Gz.T
        return x_est, P_est


    def _g(self, obs : np.ndarray, is_kinematic: bool = False, init_val: tuple = None) -> np.ndarray:
        """
            g function to transform the polar coordinate observation obs (r, theta) into local cartesian coordinates (x, y)
        """
        r, beta = obs
        g_f =  np.array([
            r * np.cos(beta),
            r * np.sin(beta)
        ])
        if is_kinematic:
            g_f = np.r_[g_f,
                        init_val[0],
                        init_val[1]
                        ]
        return g_f
    
    
    def _Gz(self, obs: np.ndarray, is_kinematic: bool = False) -> np.ndarray:
        r, beta = obs
        G_z = np.array([
            [np.cos(beta), -1. * r * np.sin(beta)],
            [np.sin(beta), r * np.cos(beta)]
        ])
        if is_kinematic:
            G_z = np.r_[
                G_z,
                np.zeros((2,2))
            ]
        return G_z

    def store_history(self, t : float, x_est : np.ndarray, P_est : np.ndarray, odo, z : dict, innov : np.ndarray, K : np.ndarray, landmarks : dict) -> None:
        if self._keep_history:
            dyn_o = self._get_tracker_hist(t)
            hist = self._htuple(
                self.description,
                t,
                x_est.copy() if x_est is not None else None,
                P_est.copy() if P_est is not None else None,
                odo.copy() if odo is not None else None,
                z.copy() if z is not None else None,
                innov.copy() if innov is not None else None,
                K.copy() if K is not None else None,
                landmarks.copy() if landmarks is not None else None,
                dyn_o if dyn_o else None,
            )
            self.history.append(hist)

    def _get_tracker_hist(self, t: float) -> dict:
        """
            Function to copy out the x_est and P_est for the trackers, because otherwise this will not work.
            returns a dict of tuples with [tracker_id] : (x_est, P_est)
            "t x_tf P_tf x_p P_p xest Pest innov K"
        """
        return {ident : self._ttuple(f"{ident}:{self.motion_model.abbreviation}",t, track.x_tf.copy(), track.P_tf.copy(), track.x_p.copy(), track.P_p.copy(), track.x_est.copy(), track.P_est.copy(), track.innovation.copy() if track.innovation is not None else None, track.K.copy() if track.K is not None else None) for ident, track in self.dyn_objects.items()} if self.dyn_objects else None
    
    def _reset_filter(self):
        """
            Function to reset the filter, such that it can be re-instantiated from scratch 
        """
        super()._reset_filter()
        self._dyn_objects = {}