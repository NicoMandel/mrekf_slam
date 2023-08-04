from typing import Tuple
from collections import namedtuple
import numpy as np
from roboticstoolbox import RangeBearingSensor
from roboticstoolbox.mobile import VehicleBase
from spatialmath import base
from mrekf.ekf_base import EKF_base
from mrekf.mr_ekf import EKF_MR
from mrekf.motionmodels import BaseModel
from mrekf.sensor import RobotSensor


class EKF_FP(EKF_MR):
    """
        EKF to include static landmarks as false positives
    """

    def __init__(self, robot, r2: list, motion_model: BaseModel, fp_list : list = None, sensor: RobotSensor = None, map=None, P0=None, x_est=None, joseph=True, animate=True, x0: np.ndarray = None, verbose=False, history=True, workspace=None, EKF_include: EKF_base = None, EKF_exclude: EKF_base = None):
        assert motion_model is not None, "Motion Model has to be specified!"
        assert fp_list is not None, "False Positive List cannot be None. Specify which landmark indices are assumed to be treated like "

        self._fp_list = fp_list

        super().__init__(robot, r2, motion_model, sensor, map, P0, x_est, joseph, animate, x0, verbose, history, workspace, EKF_include, EKF_exclude)

        self._keep_history = history  #  keep history
        if history:
            self._htuple = namedtuple("EKFlog", "t xest Pest odo z")  # todo - adapt this for logging
            self._history = []
        

    @property
    def fp_list(self) -> list:
        return self._fp_list
    
    def predict_robots(self, x_pred : np.ndarray) -> np.ndarray:
        mmsl = self.motion_model.state_length
        for r in self._fp_list:     # todo - this line the only difference. Potentially unify
            r_ind = self.robot_index(r)
            x_r = x_pred[r_ind : r_ind + mmsl]
            x_e = self.motion_model.f(x_r)
            x_pred[r_ind : r_ind + mmsl] = x_e
        return x_pred

    def split_fps(self, zk : dict) -> Tuple[dict, dict]:

        lms = {}
        rs = {}
        for lm_id, z in zk.items():
            if lm_id in self.fp_list:
                rs[lm_id] = z
            else:
                lms[lm_id] = z
        return lms, rs
    
    def step(self, t : float, odo, zk : dict) -> Tuple[np.ndarray, np.ndarray]:
        x_est = self.x_est
        P_est = self.P_est

        Fx = self.get_Fx(odo)
        Fv = self.get_Fv(odo)
        V = self.get_V()
        x_est = self.x_est
        P_est = self.P_est

        x_pred, P_pred = EKF_base.predict(x_est, P_est, self.robot, odo, Fx, Fv, V)
        x_pred = self.predict_robots(x_pred)

        #############
        # Update step -> exactly the same as mr_ekf
        #############
        zn, rk = self.split_fps(zk)
        if self.verbose:
            [print("Landmark {} seen as false positive".format(r)) for r in self.seen_robots]
        
        # split the landmarks into seen and unseen
        seen_lms, unseen_lms = self.split_readings(zn, self._isseenbefore)
        seen_rs, unseen_rs = self.split_readings(rk, self._isseenbefore_robot)
        
        # get the innovation
        innov = self.get_innovation(x_pred, seen_lms, seen_rs)
        if innov.size > 0:        
            # get the jacobians
            Hx = self.get_Hx(x_pred, seen_lms, seen_rs)
            Hw = self.get_Hw(x_pred, seen_lms, seen_rs)

            # calculate Covariance innovation, K and the rest
            x_len = int((len(x_pred) - 3) / 2)
            W_est = self.get_W_est(x_len)
            S = EKF_base.calculate_S(P_pred, Hx, Hw, W_est)
            K = EKF_base.calculate_K(Hx, S, P_pred)

            # Updating state and covariance
            x_est = EKF_base.update_state(x_pred, K, innov)
            x_est[2] = base.wrap_mpi_pi(x_est[2])
            if self._joseph:
                P_est = EKF_base.update_covariance_joseph(P_pred, K, W_est, Hx)
            else:
                P_est = EKF_base.update_covariance_normal(P_pred, S, K)
        else:
            P_est = P_pred
            x_est = x_pred
            # for history keeping
            S = None
            K = None

        # =================================================================
        # Insert New Landmarks -> exactly the same as mr_ekf
        # =================================================================
        if unseen_lms:
            W_est = self._W_est     
            n_new = len(unseen_lms) * 2
            W_est_full = self.get_W_est(int(n_new / 2))
            xf, Gz, Gx = self.get_g_funcs_lms(x_est, unseen_lms, n_new)
                
            ### section on adding the lms with the big array
            x_est, P_est = EKF_base.extend_map(
                x_est, P_est, xf, Gz, Gx, W_est_full
            )

        # inserting new robot variables
        if unseen_rs:
            W_est = self._W_est
            n_new = len(unseen_rs) * self.motion_model.state_length
            W_est_full = self.get_W_est(int(n_new / 2))
            xf, Gz, Gx = self.get_g_funcs_rs(x_est, unseen_rs, n_new)
                
            ### section on adding the lms with the big array
            x_est, P_est = EKF_base.extend_map(
                x_est, P_est, xf, Gz, Gx, W_est_full
            )

        # updating the variables before the next timestep
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
            )
            self._history.append(hist)
        return x_est, P_est