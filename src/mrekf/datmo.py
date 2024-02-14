import numpy as np
from roboticstoolbox.mobile import VehicleBase, RangeBearingSensor
from mrekf.ekf_base import BasicEKF, EKFLOG
from mrekf.motionmodels import BaseModel, KinematicModel, BodyFrame
from mrekf.ekf_math import *
from spatialmath import base

class Tracker(object):
    """
        Tracker object to be used with DATMO.
        Each dynamic landmark receives its own Tracker object
    """

    def __init__(self, ident: int,  motion_model : BaseModel, dt : float, x0 : np.ndarray, P0 : np.ndarray) -> None:
        self._mm = motion_model
        self._id = ident
        self._dt = dt
        self._x_est = x0
        self._P_est = P0

    @property
    def id(self) -> int:
        return self._id
    
    @property
    def motion_model(self) -> BaseModel:
        return self._mm
    
    @property
    def dt(self) -> float:
        return self._dt

    @property
    def x_est(self) -> np.ndarray:
        return self._x_est
    
    @property
    def P_est(self) -> np.ndarray:
        return self._P_est

    def transform(self, odo):
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
        self.x_tf = x_tf
        self.P_tf = P_tf

        return x_tf, P_tf 

    def predict(self):
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
        return x_pred, P_pred
    
    def update(self, obs):
        inn = self.get_innovation(obs)

class DATMO(BasicEKF):

    def __init__(self, description : str, 
                 dynamic_ids: list, motion_model : BaseModel,  x0: np.ndarray = np.array([0., 0., 0.]), P0: np.ndarray = None, robot: tuple[VehicleBase, np.ndarray] = None, sensor: tuple[RangeBearingSensor, np.ndarray] = None,
                history: bool = False, joseph: bool = True, ignore_ids: list = []) -> None:
        super().__init__(description, x0, P0, robot, sensor, history, joseph, ignore_ids)
        self._dynamic_ids = dynamic_ids
        self._motion_model = motion_model

        # adding dynamic objects
        self._dyn_objects = {}

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
    def dynamic_ids(self) -> set:
        return self._dynamic_ids

    @property
    def seen_dyn_lms(self) -> set:
        return set(self.landmarks.keys()).intersection(set(self.dynamic_ids))

    @property
    def seen_static_lms(self) -> set:
        return set(self.landmarks.keys()) - set(self.dynamic_ids)

    @property
    def dyn_objects(self) -> dict:
        return self._dyn_objects

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

    # Step has to be overwritten, because the transformation needs the odometry
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
        x_pred, P_pred = self.predict(odo, x_est, P_est)

        # split readings into seen and unseen and ignore the ones that are supposed to be ignored.
        seen, unseen = self.process_readings(zk)

        # update - changes made here!
        x_est, P_est, innov, K = self.update(x_pred, P_pred, seen, odo)

        # insert new things
        x_est, P_est = self.extend(x_est, P_est, unseen)

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
            self.dyn_objects[ident].transform(odo)
            self.dyn_objects[ident].predict()
            
            # Todo -> sensor model is used to create Hx, Hp and innovation.
            self.dyn_objects[ident].update(obs)

        return x_est, P_est, innov, K

    def extend(self, x_est : np.ndarray, P_est : np.ndarray, unseen : dict) -> tuple[np.ndarray, np.ndarray]:
        """

        """
        # 1. split the observations into dynamic and static observations
        static, dynamic = self.split_observations(unseen)
        
        # 2. insert the static observations
        x_est, P_est = super().extend(x_est, P_est, static)

        # 3. create a new tracker object for each dynamic landmark and add it to the dictionary
        for ident, obs in dynamic.items():
            # 4. TODO: create Yz and G according to the simulation from PC 6.2 or 6.3
            x_n, P_n = self.init_dyn(obs)
            # 5. create a tracker and insert it
            nt = Tracker(ident, self.motion_model, self.robot.dt, x_n, P_n)
            
            self.dyn_objects[ident] = nt

        return x_est, P_est
    
    def init_dyn(self, obs) -> tuple[np.ndarray, np.ndarray]:
        """
            TODO:
            function to create new object from observations
            Get from PC - inserting with known pose 
        """



    # TODO - overwrite this!
    def store_history(self, t : float, x_est : np.ndarray, P_est : np.ndarray, odo, z : dict, innov : np.ndarray, K : np.ndarray, landmarks : dict) -> None:
        """
            
        """
        if self._keep_history:
            hist = self._htuple(
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
