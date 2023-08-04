from typing import Tuple
import numpy as np
from roboticstoolbox import RangeBearingSensor
from roboticstoolbox.mobile import VehicleBase
from mrekf.ekf_base import EKF_base
from mrekf.mr_ekf import EKF_MR
from mrekf.motionmodels import BaseModel


class EKF_FP(EKF_base, EKF_MR):
    """
        EKF to include static landmarks as false positives
        todo can theoretically rerun the step phase and just overwrite the following functions:
            * predict static
            * update_static
            * extend_static
    """

    def __init__(self, x0: np.ndarray = None, P0: np.ndarray = None, robot: VehicleBase = None, motion_model : BaseModel = None, fp_list : list = None, sensor: RangeBearingSensor = None, history: bool = False, joseph: bool = True) -> None:
        
        assert motion_model is not None, "Motion Model has to be specified!"
        assert fp_list is not None, "False Positive List cannot be None. Specify which landmark indices are assumed to be treated like "

        self._fp_list = fp_list
        super().__init__(x0, P0, robot, sensor, history, joseph)

    @property
    def fp_list(self) -> list:
        return self._fp_list
    
    def predict(self, x_pred : np.ndarray, P_pred : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            prediction function when assuming FPs
        """
        pass

    def update(self, x_pred : np.ndarray, P_pred : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            Update function when assuming FPs
        """
        pass

    def extend(self, x_pred : np.ndarray, P_pred : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            Extending a map when assuming FPs
        """
        pass

    def step(self, t : float, odo, zk : dict) -> Tuple[np.ndarray, np.ndarray]:
        x_est = self.x_est
        P_est = self.P_est

        x_pred, P_pred = self.predict(x_est, P_est)

        seen, unseen = self.split_readings(zk)

        x_est, P_est = self.update(x_pred, P_pred, seen)

        x_est, P_est = self.extend(x_est, P_est, unseen)

        return x_est, P_est