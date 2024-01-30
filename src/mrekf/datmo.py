import numpy as np
from roboticstoolbox.mobile import VehicleBase, RangeBearingSensor
from mrekf.ekf_base import BasicEKF, EKFLOG
from mrekf.motionmodels import BaseModel, KinematicModel, BodyFrame
from mrekf.ekf_math import extend_map, np
from spatialmath import base

class Tracker(object):
    """
        Tracker object to be used with DATMO.
        Each dynamic landmark receives its own Tracker object
    """

    def __init__(self, ident: int,  motion_model : BaseModel, dt : float) -> None:
        self._mm = motion_model
        self._id = ident
        self._dt = dt

    @property
    def id(self) -> int:
        return self._id
    
    @property
    def motion_model(self) -> BaseModel:
        return self._mm
    
    @property
    def dt(self) -> float:
        return self._dt

    def predict(self):
        pass

    def transform(self, x_est : np.ndarray, P_est : np.ndarray):
        """
            transform requires an increase in uncertainty, see Sola p. 154
        """
        pass
    
    def update(self):
        pass

    def j(self, x_est : np.ndarray,  odo) -> np.ndarray:
        """
            Function j(O, u) from Sola p. 154, which casts the current object state in the robot frame at time k into the robot frame at k_+1
            odo is in the form v, theta (rad)
        """
        v, theta = odo
        R = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        t = np.asarray([v * self.dt, 0.])
        x_pred = R.T @ (x_est - t[:,np.newaxis])
        return x_pred
    
    def Jo(self, x_est : np.ndarray, odo) -> np.ndarray:
        """
            Function J_o for transformation of P_pred.
            Derivative of frame transformation for measurement wrt to previous object states x_k and y_k
            See Sola p.154
        """
        _, theta = odo
        Jo = np.asarray([
            [np.cos(theta), np.sin(theta)],
            [-1. * np.sin(theta), np.cos(theta)]
        ])
        return Jo
    
    def Ju(self, x_est : np.ndarray, odo) -> np.ndarray:
        """
            Function J_u for transformation of P_pred.
            Derivative of frame transformation for measurement wrt. to Input Vector u for Robot motion model 
            See Sola p.154
        """
        v, theta = odo
        Ju = np.ndarray([
            [],
            []
        ])
        return Ju
    

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
    
    @property
    def state_length(self) -> int:
        l = 3 + 2 * len(self.seen_static_lms)
        return l
    
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

    def predict_x(self, x_est : np.ndarray, odo) -> tuple[np.ndarray, np.ndarray]:
        """
            Prediction function for x. should perform prediction of the robot itself normally
        """
        x_pred = super().predict_x(x_est, odo)
        
        # prediction in the **current** robot frame for each of its dynamic landmarks.

        return x_pred

    def update(self, x_pred : np.ndarray, P_pred : np.ndarray, seen : dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            Update function overwritten for DATMO. Updating the dynamic landmarks is done after the state update, with the new state.
        """
        # 1. split the observations into dynamic and static observations
        static, dynamic = self.split_observations(seen)
        # 2. perform the regular update with only the static landmarks
        x_est, P_est, innov, K = super().update(x_pred, P_pred, static)

        # 3. for each dynamic landmark - transform into new frame and update
        for ident, obs in dynamic.items():
            pass

        return x_est, P_est, innov, K

    def extend(self, x_est : np.ndarray, P_est : np.ndarray, unseen : dict) -> tuple[np.ndarray, np.ndarray]:
        """

        """
        # 1. split the observations into dynamic and static observations
        static, dynamic = self.split_observations(unseen)
        # 2. insert the static observations
        x_est, P_est = super().extend(x_est, P_est)
        # 3. create a new tracker object for each dynamic landmark and add it to the dictionary
        for ident, obs in dynamic.items():
            pass

        return x_est, P_est