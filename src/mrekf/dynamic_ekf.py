import numpy as np
from roboticstoolbox.mobile import VehicleBase
from mrekf.ekf_base import BasicEKF, MR_EKFLOG
from mrekf.motionmodels import BaseModel

class Dynamic_EKF(BasicEKF):

    def __init__(self, dynamic_ids: list, motion_model : BaseModel,  x0: np.ndarray = np.array([0., 0., 0.]), P0: np.ndarray = None, robot: tuple[VehicleBase, np.ndarray] = None, W: np.ndarray = None, history: bool = False, joseph: bool = True, ignore_ids: list = []) -> None:
        super().__init__(x0, P0, robot, W, history, joseph, ignore_ids)
        self._dynamic_ids = dynamic_ids
        self._motion_model = motion_model

        # overwriting history keeping!
        if history:
            self._htuple = MR_EKFLOG
            self._history = []

    @property
    def dynamic_idcs(self) -> set:
        return self._dynamic_ids

    @property
    def seen_dyn_lms(self) -> set:
        return set(self.landmarks.keys()).intersection(set(self.dynamic_idcs))

    @property
    def seen_static_lms(self) -> set:
        return set(self.landmarks.keys()) - set(self.dynamic_idcs)

    @property
    def motion_model(self) -> list:
        return self._motion_model
    
    @property
    def state_length(self):
        """
            Todo - could change this by storing at runtime whether somethings is dynamic or not
        """
        l = 3 + 2 * len(self.seen_dyn_lms) + self.motion_model.state_length * len(self.seen_dyn_lms)
        return l

    def store_history(self, t: float, x_est: np.ndarray, P_est: np.ndarray, z: dict, innov: np.ndarray, K: np.ndarray, landmarks: dict) -> None:
        """
        """
        if self._keep_history:
            hist = self._htuple(
                # todo insert values we are storing here!
            )

    def _get_Fx(self, x_est: np.ndarray, odo) -> np.ndarray:
        Fx = super()._get_Fx(x_est, odo)
        mmsl = self.motion_model.state_length
        for r in self.seen_dyn_lms:
            
