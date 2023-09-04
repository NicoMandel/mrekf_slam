import numpy as np
from roboticstoolbox.mobile import VehicleBase
from mrekf.ekf_base import BasicEKF, MR_EKFLOG
from mrekf.ekf_math import np
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
    def motion_model(self) -> BaseModel:
        return self._motion_model
    
    @property
    def state_length(self):
        """
            Todo - check if works
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

    def predict_x(self, x_est : np.ndarray, odo) -> np.ndarray:
        """
            Overwritten -> needs to incorporate the motion model
            ! this is happening inplace - maybe should copy x_est?
        """
        x_pred = x_est.copy()
        xv_est = x_pred[:3]
        xv_pred = self.robot.f(xv_est, odo)
        x_pred[:3] = xv_pred
        
        mmsl = self.motion_model.state_length
        for dlm in self.seen_dyn_lms:
            d_ind = self.landmark_index(dlm)
            x_d = x_est[d_ind : d_ind + mmsl]
            x_e = self.motion_model.f(x_d)
            x_est[d_ind : d_ind + mmsl] = x_e 
        return x_est
        

    def _get_Fx(self, x_est: np.ndarray, odo) -> np.ndarray:
        Fx = super()._get_Fx(x_est, odo)
        mmsl = self.motion_model.state_length
        for dlm in self.seen_dyn_lms or []:
            d_ind = self.landmark_index(dlm)
            xd = x_est[d_ind : d_ind + mmsl]
            Fx[d_ind : d_ind + mmsl, d_ind : d_ind + mmsl] = self.motion_model.Fx(xd)

        return Fx

    def _get_Fv(self, x_est: np.ndarray, odo) -> np.ndarray:
        Fv =  super()._get_Fv(x_est, odo)
        mmsl = self.motion_model.state_length
        for dlm in self.seen_dyn_lms or []:
            d_ind = self.landmark_index(dlm)
            xd = x_est[d_ind : d_ind + mmsl]
            ins_r = d_ind
            ins_c = d_ind - 1
            Fv[ins_r : ins_r + mmsl, ins_c : ins_c + mmsl] = self.motion_model.Fv(xd)
        
        return Fv
    
    def _get_V(self, x_est: np.ndarray) -> np.ndarray:
        Vm =  super()._get_V(x_est)
        mmsl = self.motion_model.state_length
        for dlm in self.seen_dyn_lms or []:
            d_ind = self.landmark_index(dlm) - 1
            Vm[d_ind : d_ind + mmsl, d_ind : d_ind + mmsl] = self.motion_model.V
        
        return Vm
