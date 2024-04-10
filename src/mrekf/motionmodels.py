
import numpy as np
from abc import ABC, abstractmethod
from spatialmath import base
from scipy.linalg import block_diag
from roboticstoolbox.mobile import VehicleBase
import spatialmath.base as smb

from mrekf.transforms import forward, inverse, pol2cart

class BaseModel(ABC):
    """        
    """

    @abstractmethod
    def __init__(self, V : np.ndarray, dt : float, Fv : np.ndarray, Fx : np.ndarray, state_length : int) -> None:
        self._V = V
        self._Fv = Fv
        self._Fx = Fx
        self._state_length = state_length
        self._dt = dt
    
    def __str__(self):
        # s = super().__str__()
        s = f"{self.__class__.__name__} motion model\n"
        s += f"\n  V = {base.array2str(self.V)}\n"
        if hasattr(self, "dt"):
            s += f"  dt: ({self.dt})\n"
        return s.rstrip()

    def __repr__(self) -> str:
        return str(self)
    
    @property
    @abstractmethod
    def abbreviation(self) -> str:
        """
            Abbreviation used to get the model type
        """
        pass

    # Motion Models
    @abstractmethod
    def f(self, x : np.ndarray) -> np.ndarray:
        pass
    
    def Fv(self, x : np.ndarray = None) -> np.ndarray:
        return self._Fv

    def Fx(self, x : np.ndarray = None) -> np.ndarray:
        return self._Fx

    @abstractmethod
    def get_true_state(self) -> None:
        pass

    @property
    def state_length(self) -> int:
        return self._state_length

    @property
    def V(self) -> np.ndarray:
        return self._V
    
    @property
    def dt(self) -> float:
        return self._dt

    def scale_V(self, scale : float = None) -> np.ndarray:
        return scale * self.V if scale is not None else self.V
    
    # j function and Jacobians Jo and Ju for reframing as defined in Sola p. 154
    # Defined as functions of the superclass because the base point needs to be reframed all the time, and KinematicModel and BodyFrameModel only overwrite the second parts
    def j(self, x : np.ndarray, odo, dtheta : float) -> np.ndarray:
        """
            Function j for reframing. 
        """
        v = odo[0]
        t = np.asarray([
            self.dt * v,     # + instead of minus!
            0.,
            dtheta
        ])
        # actual calculation
        x_n = inverse(t, x)
        return x_n

    def Jo(self, x : np.ndarray, odo : tuple, dtheta : float) -> np.ndarray:
        """
            Jacobian of j(o,u) wrt previous state Jo for converting the covariances
        """
        Jo = smb.rot2(dtheta).T
        return Jo
    
    def Ju(self, x : np.ndarray, odo : tuple, dtheta : float) -> np.ndarray:
        """
            Jacobian of j(o,u) wrt to roboto control inputes Ju for converting the covariances
        """
        v = odo[0]
        f1 = self.dt * v - x[0]
        Ju = np.asarray([
            [-1. * self.dt * np.cos(dtheta), np.sin(dtheta) * f1 + x[1] * np.cos(dtheta)],
            [self.dt * np.sin(dtheta), np.cos(dtheta) * f1 - x[1] * np.sin(dtheta)]
        ])
        return Ju


class StaticModel(BaseModel):
    """
        Class to provide a static motion model.
        Including derivatives
        defined as 
        :math:  x_k+1 = x_k + v_x
        :math:  y_k+1 = y_k + v_y
    """

    def __init__(self, V : np.ndarray, dt : float) -> None:
        assert V.shape == (2,2), "V not correct shape, Please make sure it's 2x2"
        dim = 2 

        # derivative matrices
        Fv = np.eye(dim, dtype=float)
        Fx = np.eye(dim, dtype=float)
        super().__init__(V, dt, Fv, Fx, dim)      
    
    def f(self, x : np.ndarray) -> np.ndarray:
        """
            f(x_k+1) = x_k + v_x
        """
        return x
    
    @property
    def abbreviation(self) -> str:
        return "SM"
    
    def get_true_state(self) -> None:
        return None

class KinematicModel(BaseModel):
    """
        Class to provide a kinematic motion model
        defined as 
        :math:  x_k+1 = x_k + dt * (x_k_dot + v_x)
        :math:  y_k+1 = y_k + dt * (y_k_dot + v_y)
        :math:  y_k_dot+1 = y_k_dot + v_y
        :math:  x_k_dot+1 = x_k_dot + v_x
        Using help from: # see https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/
    """

    def __init__(self, V : np.ndarray, dt : float, vmax : float = 3.0) -> None:
        assert V.shape == (2,2), "V not correct shape, Please make sure it's 2x2"
        dim = 4

        # derivative Matrices
        Fx = np.array([
            [1., 0., dt, 0.],
            [0., 1., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])

        Fv = np.array([
            [dt, 0.],
            [0., dt],
            [1., 0.],
            [0., 1.],
        ])
        super().__init__(V, dt, Fv, Fx, dim)
        self._vmax = (vmax, vmax)

        # A matrix for state prediction
        self._A = np.array([
            [1., 0., dt, 0.],
            [0., 1., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])
    
    @property
    def abbreviation(self) -> str:
        return "KM"

    @property
    def vmax(self) -> tuple:
        return self._vmax

    @property
    def A(self) -> np.ndarray:
        return self._A

    def get_true_state(self, v : float, theta : float) -> tuple:
        """
            Function to get the true state of the hidden parts from a robot. For initialisation
        """
        v_xy = pol2cart((v, theta))
        return v_xy[0], v_xy[1]       

    def f(self, x : np.ndarray) -> np.ndarray:
        fx_k = self.A @ x
        return fx_k   

    # function for reframing j and Jacobians Jo and Ju
    def j(self, x : np.ndarray, odo : tuple, dtheta : float) -> np.ndarray:
        """
            Inherits from parent.
            Adds velocity transformation through rotation matrix
        """
        xy = x[:2]
        xy_dot = x[2:]
        j_xy = super().j(xy, odo, dtheta)

        R = smb.rot2(dtheta)
        j_xy_dot = R.T @ xy_dot     # inverse rotations
        j_f = np.r_[
            j_xy,
            j_xy_dot
        ]
        return j_f

    def Jo(self, x: np.ndarray, odo : tuple, dtheta : float) -> np.ndarray:
        """
            Inherits from parent.
            Adds velocity transformation through rotation matrix derivative.
            Is 4x4 matrix. 4 functions, 4 states
        """
        xy = x[:2]
        Jo_xy = super().Jo(xy, odo, dtheta)

        Jo_xy_dot = smb.rot2(dtheta).T
        Jo_f = block_diag(Jo_xy, Jo_xy_dot)
        return Jo_f
        
    def Ju(self, x: np.ndarray, odo : tuple, dtheta : float) -> np.ndarray:
        """
            Inherits from parent.
            Adds velocity transformation through rotation matrix derivative multiplied by -1.
            since velocities are independent of other velocities in the frame transformation, the first indices are 0.
            is a 4x2 matrix. 4 functions, 2 states.
        """
        xy = x[:2]
        Ju_xy =  super().Ju(xy, odo, dtheta)
        xy_dot = x[2:]

        R_alt = -1. * np.asarray([
            [np.sin(dtheta), -1. * np.cos(dtheta)],
            [np.cos(dtheta), np.sin(dtheta)]
        ])
        # Jo_xy_dot = smb.rot2(dtheta)
        Ju_xy_dot_theta = R_alt @ xy_dot
        Ju_xy_dot = np.c_[
            np.zeros((2,1)),
            Ju_xy_dot_theta
        ]
        Ju_f = np.r_[
            Ju_xy,
            Ju_xy_dot
        ]
        return Ju_f

class BodyFrame(BaseModel):
    """
        Class to provide a Body Frame model, similar to the bicycle model used by PC. However, we include v and omega in the state.
        So the state is:
            - x
            - y
            - v
            - omega 
        changing the state update equations!
    """

    def __init__(self, V : np.ndarray, dt : float, vmax : tuple = (3.0, 0.01)):
        assert V.shape == (2,2), "V not correct shape, Please make sure it's 2x2"
        dim = 4

        Fv = None
        Fx = None
        super().__init__(V, dt, Fv, Fx, dim)

        self._vmax = vmax

    @property
    def abbreviation(self) -> str:
        return "BF"

    @property
    def vmax(self) -> tuple:
        return self._vmax

    def f(self, x: np.ndarray) -> np.ndarray:
        """
            Model to predict forward
        """
        x_k = np.array([
            x[0] + self.dt * x[2] * np.cos(x[3]),
            x[1] + self.dt * x[2] * np.sin(x[3]), 
            x[2],
            base.wrap_mpi_pi(x[3])
        ])
        return x_k
    
    def Fx(self, x : np.ndarray) -> np.ndarray:
        """
            Since these are nonlinear models, the update equations are different.
            # todo check angle wrapping!
        """
        cos_x3 = np.cos(base.wrap_mpi_pi(x[3] + self.V[1,1]))
        sin_x3 = np.sin(base.wrap_mpi_pi(x[3] + self.V[1,1]))
        fx = np.array([
            [1., 0., self.dt * cos_x3,  -1. * self.dt * (x[2] + self.V[0,0]) * sin_x3],
            [0., 1., self.dt * sin_x3,   self.dt * (x[2] + self.V[0,0]) * cos_x3],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
            ])
        return fx

    def Fv(self, x: np.ndarray) -> np.ndarray:
        cos_x3 = np.cos(base.wrap_mpi_pi(x[3] + self.V[1,1]))
        sin_x3 = np.sin(base.wrap_mpi_pi(x[3] + self.V[1,1]))
        fv = np.array([
            [self.dt * cos_x3,  -1. * self.dt * (x[2] + self.V[0,0]) * sin_x3],
            [self.dt * sin_x3,  self.dt * (x[2] + self.V[0,0]) * cos_x3],
            [1., 0.],
            [0., 1.]
            ])
        return fv

    def get_true_state(self, v : float, theta : float) -> tuple:
        """
            Function to get the true state of the hidden parts from a robot. For initialisation
        """
        return v, theta   
    
    # functions for reframing j and Jacobians Jo and Ju
    def j(self, x: np.ndarray, odo : tuple, dtheta: float) -> np.ndarray:
        xy = x[:2]
        xy_n = super().j(xy, odo, dtheta)
        xy_bf = x[2:]

        x_n = np.r_[
            xy_n,
            xy_bf[0],
            xy_bf[1] - dtheta   # careful -> minus here
        ]
        return x_n

    def Jo(self, x: np.ndarray, odo : tuple, dtheta : float) -> np.ndarray:
        xy = x[:2]
        Jo_xy = super().Jo(xy, odo, dtheta)
        Jo_vt = np.eye(2)
        Jo_n = block_diag(Jo_xy, Jo_vt)
        return Jo_n
    
    def Ju(self, x: np.ndarray, odo : tuple, dtheta : float) -> np.ndarray:
        xy = x[:2]
        Ju_xy =  super().Ju(xy, odo, dtheta)
        
        Ju_vt = np.zeros((2,2))
        Ju_vt[1,1] = -1.

        Ju_n = np.r_[
            Ju_xy,
            Ju_vt
        ]

        return Ju_n

