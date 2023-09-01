import numpy as np
from roboticstoolbox.mobile import VehicleBase
from mrekf.ekf_base import BasicEKF, MR_EKFLOG
from mrekf.ekf_math import VehicleBase
from mrekf.motionmodels import BaseModel



class Dynamic_EKF(BasicEKF):

    def __init__(self, dynamic_ids: list, motion_model : BaseModel,  x0: np.ndarray = np.array([0., 0., 0.]), P0: np.ndarray = None, robot: tuple[VehicleBase, np.ndarray] = None, W: np.ndarray = None, history: bool = False, joseph: bool = True, ignore_ids: list = []) -> None:
        super().__init__(x0, P0, robot, W, history, joseph, ignore_ids)
        self._dynamic_ids = dynamic_ids
        self._motion_model = motion_model

    @property
    def dynamic_idcs(self) -> list:
        return self._dynamic_ids
    
    @property
    def motion_model(self) -> list:
        return self._motion_model
    
    
    

