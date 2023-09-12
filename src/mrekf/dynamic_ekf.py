import numpy as np
from roboticstoolbox.mobile import VehicleBase, RangeBearingSensor
from mrekf.ekf_base import BasicEKF, MR_EKFLOG
from mrekf.motionmodels import BaseModel
from mrekf.ekf_math import extend_map
from spatialmath import base
# ! check call order - if super.step() is called with ref to predict_x() in child class, will this call child predict_x() or parent?


class Dynamic_EKF(BasicEKF):

    def __init__(self, dynamic_ids: list, motion_model : BaseModel,  x0: np.ndarray = np.array([0., 0., 0.]), P0: np.ndarray = None, robot: tuple[VehicleBase, np.ndarray] = None, sensor: tuple[RangeBearingSensor, np.ndarray] = None,
                history: bool = False, joseph: bool = True, ignore_ids: list = []) -> None:
        super().__init__(x0, P0, robot, sensor, history, joseph, ignore_ids)
        self._dynamic_ids = dynamic_ids
        self._motion_model = motion_model

        # overwriting history keeping!
        if history:
            self._htuple = MR_EKFLOG
            self._history = []

    def __str__(self):
        s = f"{self.__class__.__name__} object: {len(self._x_est)} states"

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

    def __repr__(self) -> str:
        return str(self)

    @property
    def dynamic_ids(self) -> set:
        return self._dynamic_ids

    @property
    def seen_dyn_lms(self) -> set:
        return set(self.landmarks.keys()).intersection(set(self.dynamic_ids))

    @property
    def seen_static_lms(self) -> set:
        return set(self.landmarks.keys()) - set(self.dynamic_ids)

    def dynamic_lms_in_dict(self, sd : dict) -> dict:
        return {k : v for k, v in sd.items() if k in self.dynamic_ids}

    def static_lms_in_dict(self, sd : dict) -> dict:
        return {k : v for k, v in sd.items() if k not in self.dynamic_ids}

    @property
    def motion_model(self) -> BaseModel:
        return self._motion_model
    
    @property
    def state_length(self):
        """
            Todo - check if works
        """
        l = 3 + 2 * len(self.seen_static_lms) + self.motion_model.state_length * len(self.seen_dyn_lms)
        return l

    def has_kinematic_model(self) -> bool:
        return True if self.motion_model.state_length > 2 else False

    def store_history(self, t: float, x_est: np.ndarray, P_est: np.ndarray, z: dict, innov: np.ndarray, K: np.ndarray, landmarks: dict) -> None:
        """
        """
        if self._keep_history:
            hist = self._htuple(
                # todo insert values we are storing here!
            )

    # Overwriting necessary prediction functions
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

    # Overwriting necessary update functions
    def get_Hx(self, x_pred: np.ndarray, seen: dict) -> np.ndarray:
        dim = x_pred.size
        Hx = np.zeros((dim-3, dim))
        xv_pred = x_pred[:3]

        for lm_id, _ in seen.items():
            lm_ind = self.landmark_index(lm_id)
            xf = x_pred[lm_ind : lm_ind + 2]
            # treat dynamic
            if lm_id in self.dynamic_ids:
                dyn = True
                mmsl = self.motion_model.state_length
            # treat static
            else:
                dyn = False
                mmsl = 2
            
            Hp_k = self.sensor.Hp(xv_pred, xf, dyn)
            Hxv = self.sensor.Hx(xv_pred, xf)

            lm_mind = lm_ind - 3
            Hx[lm_mind : lm_mind+2, :3] = Hxv
            Hx[lm_mind : lm_mind + 2, lm_mind : lm_mind + mmsl] = Hp_k

        return Hx
    
    # Overwriting necessary extending functions
    def extend(self, x_est: np.ndarray, P_est: np.ndarray, unseen: dict) -> tuple[np.ndarray, np.ndarray]:
        dyn_lms = self.dynamic_lms_in_dict(unseen)
        stat_lms = self.static_lms_in_dict(unseen)        
        n_new = len(stat_lms) * 2 + len(dyn_lms) * self.motion_model.state_length
        W_est_full = self.get_W_est(int(n_new / 2))
        xf, Gz, Gx = self.get_g_funcs(x_est, unseen, n_new) 
        
        x_est, P_est = extend_map(
            x_est, P_est, xf, Gz, Gx, W_est_full
        )
        return x_est, P_est

    def get_g_funcs(self, x_est: np.ndarray, unseen: dict, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        Gx = np.zeros((n, 3))
        Gz = np.zeros((n, n))
        xf = np.zeros(n)

        xv = x_est[:3]
        start_ind = 0
        for lm_id, z in unseen.items():
            if lm_id in self.dynamic_ids:
                mmsl = self.motion_model.state_length
                dyn = True
            else:
                mmsl = 2
                dyn = False
            
            xf_i = self.sensor.g(xv, z, dyn)
            Gz_i = self.sensor.Gz(xv, z, dyn)
            Gx_i = self.sensor.Gx(xv, z, dyn)

            xf[start_ind : start_ind + mmsl] = xf_i
            Gz[start_ind : start_ind + mmsl, start_ind : start_ind + 2] = Gz_i
            Gx[start_ind : start_ind + mmsl, :] = Gx_i

            start_ind += mmsl

            self._landmark_add(lm_id)

        return xf, Gz, Gx