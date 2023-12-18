import numpy as np
from roboticstoolbox.mobile import VehicleBase, RangeBearingSensor
from mrekf.ekf_base import BasicEKF, EKFLOG
from mrekf.motionmodels import BaseModel
from mrekf.ekf_math import extend_map, np
from spatialmath import base
# ! check call order - if super.step() is called with ref to predict_x() in child class, will this call child predict_x() or parent?


class Dynamic_EKF(BasicEKF):

    def __init__(self, description : str, 
                 dynamic_ids: list, motion_model : BaseModel,  x0: np.ndarray = np.array([0., 0., 0.]), P0: np.ndarray = None, robot: tuple[VehicleBase, np.ndarray] = None, sensor: tuple[RangeBearingSensor, np.ndarray] = None,
                history: bool = False, joseph: bool = True, ignore_ids: list = []) -> None:
        super().__init__(description, x0, P0, robot, sensor, history, joseph, ignore_ids)
        self._dynamic_ids = dynamic_ids
        self._motion_model = motion_model

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

    @property
    def is_dynamic(self) -> bool:
        return True

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
    def state_length(self) -> int:
        """
            Todo - check if works
        """
        l = 3 + 2 * len(self.seen_static_lms) + self.motion_model.state_length * len(self.seen_dyn_lms)
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
            x_pred[d_ind : d_ind + mmsl] = x_e 
        return x_pred
        
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
            
            # row corresponds to the index in the state vector
            ins_r = d_ind
            
            # column where to insert is 2 * (1 + number of landmarks seen before)
            lm_pos = self.landmark_position(dlm)
            ins_c = 2 * (1+ lm_pos)
            Fv[ins_r : ins_r + mmsl, ins_c : ins_c + 2] = self.motion_model.Fv(xd)
        
        return Fv
    
    def _get_V(self) -> np.ndarray:
        """
            Getting a large V - matrix. Super is a matrix of 0s, only with index of vehicle filled.
            The indices in the lm pos are filled with the v from the motionn model
        """
        Vm =  super()._get_V()
        # mmsl = self.motion_model.state_length
        for dlm in self.seen_dyn_lms or []:
            lm_pos = self.landmark_position(dlm)
            d_ind = 2 * (1 + lm_pos)
            # d_ind = self.landmark_index(dlm) - 1
            Vm[d_ind : d_ind + 2, d_ind : d_ind + 2] = self.motion_model.V
        
        return Vm
    
    def _get_V_alt(self) -> np.ndarray:
        """
            Use of Kronecker product to get the V- noise matrix.
            this version could theoretically be included into the ekf-base version. because V is only used for Fv predict step.
            But doesn't work, because _V depends on the motion model -> therefore ekf_base has no knowledge of this.
            this would turn into a simple matrix only where there are dynamic landmarks
        """
        dim = (1 + len(self.landmarks))
        _V = self.motion_model.V
        Vm = np.kron(np.eye(dim), _V)
        V_v = self.V_est        
        Vm[:2, :2] = V_v
        return Vm

    
    def get_Hx(self, x_pred: np.ndarray, seen: dict) -> np.ndarray:
        """
            Overwritten. 
        """
        # size is the same
        cols = x_pred.size
        rows = 2 * len(seen)
        Hx = np.zeros((rows, cols))
        xv_pred = x_pred[:3]

        # have to ensure to skip the additional columns for the states - otherwise distributes wrong
        start_row = 0
        for lm_id in seen:
            # get the predicted lm values
            s_ind = self.landmark_index(lm_id)
            xf = x_pred[s_ind : s_ind + 2]

            # decide if the lm is treated as dynamic or static
            if lm_id in self.dynamic_ids:
                mmsl = self.motion_model.state_length
                dyn = True
            else:
                mmsl = 2
                dyn = False
            
            # do the sensor prediction
            Hp_k = self.sensor.Hp(xv_pred, xf, dyn)
            Hxv = self.sensor.Hx(xv_pred, xf)
            
            # insert the values into the corresponding positions
            Hx[start_row : start_row + 2, :3] = Hxv
            Hx[start_row : start_row + 2, s_ind : s_ind + mmsl] = Hp_k

            start_row += 2

        return Hx

    def get_g_funcs(self, x_est: np.ndarray, unseen: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dyn_lms = self.dynamic_lms_in_dict(unseen)
        stat_lms = self.static_lms_in_dict(unseen)        
        n = len(stat_lms) * 2 + len(dyn_lms) * self.motion_model.state_length

        # construct the G-Matrices 
        Gx = np.zeros((n, 3))
        Gz = np.zeros((n, 2 * len(unseen)))
        xf = np.zeros(n)

        xv = x_est[:3]
        start_ind = 0
        start_col = 0

        for lm_id, z in unseen.items():
            if lm_id in self.dynamic_ids:
                mmsl = self.motion_model.state_length
                dyn = True
            else:
                mmsl = 2
                dyn = False
            
            # Get the landmark values from the sensor model
            xf_i = self.sensor.g(xv, z, dyn)
            Gz_i = self.sensor.Gz(xv, z, dyn)
            Gx_i = self.sensor.Gx(xv, z, dyn)
            
            # fill in the large matrices
            xf[start_ind : start_ind + mmsl] = xf_i
            Gz[start_ind : start_ind + mmsl, start_col : start_col + 2] = Gz_i
            Gx[start_ind : start_ind + mmsl, :] = Gx_i

            start_ind += mmsl
            start_col += 2

            self._landmark_add(lm_id)

        return xf, Gz, Gx