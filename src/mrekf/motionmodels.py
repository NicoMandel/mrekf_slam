
import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def __init__(self, V : np.ndarray, Fv : np.ndarray, Fx : np.ndarray, state_length : int) -> None:
        self._V = V
        self._Fv = Fv
        self._Fx = Fx
        self._state_length = state_length
    
    # Motion Models
    @abstractmethod
    def f(self, x : np.ndarray) -> np.ndarray:
        pass
    
    def Fv(self, x : np.ndarray = None) -> np.ndarray:
        return self._Fv

    def Fx(self, x : np.ndarray = None) -> np.ndarray:
        return self._Fx

    @property
    def state_length(self) -> int:
        return self._state_length

    @property
    def V(self) -> np.ndarray:
        return self._V
    
    def scale_V(self, scale : float = None) -> np.ndarray:
        return scale * self.V if scale is not None else self.V
    
class StaticModel(BaseModel):
    """
        Class to provide a static motion model.
        Including derivatives
        defined as 
        :math:  x_k+1 = x_k + v_x
        :math:  y_k+1 = y_k + v_y
    """

    def __init__(self, V : np.ndarray) -> None:
        assert V.shape == (2,2), "V not correct shape, Please make sure it's 2x2"
        dim = 2 

        # derivative matrices
        Fv = np.eye(dim, dtype=float)
        Fx = np.eye(dim, dtype=float)
        super().__init__(V, Fv, Fx, dim)      
    
    def f(self, x : np.ndarray) -> np.ndarray:
        """
            f(x_k+1) = x_k + v_x
        """
        return x

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
        assert V.shape == (4,4), "V not correct shape, Please make sure it's 4x4"
        dim = 4

        # derivative Matrices
        Fx = np.array([
            [1., 0., dt, 0.],
            [0., 1., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])

        Fv = np.array([
            [0., 0., dt, 0.],
            [0., 0., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
        super().__init__(V, Fv, Fx, dim)
        self._dt = dt
        self._vmax = vmax

        # A matrix for state prediction
        self._A = np.array([
            [1., 0., dt, 0.],
            [0., 1., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])
        
    @property
    def dt(self) -> float:
        return self._dt

    @property
    def vmax(self) -> float:
        return self._vmax

    @property
    def A(self) -> np.ndarray:
        return self._A


    def f(self, x : np.ndarray) -> np.ndarray:
        fx_k = self.A @ x
        return fx_k   

class BodyFrame(BaseModel):
    """
        Class to provide a Body Frame model, similar to the bicycle model used by PC. However, we include v and omega in the state.
        So the state is:
            - x
            - y
            - v
            - omega # todo - check omega has all the base_wrap variables
        changing the state update equations!
    """

    def __init__(self, V : np.ndarray, dt : float, vmax : float = 3.0):
        assert V.shape == (4,4), "V not correct shape, Please make sure it's 4x4"
        assert np.allclose(V[:2,:2], np.zeros((2,2))), "The top left corner of V should be all zeros, please correct"
        dim = 4

        Fv = None
        Fx = None
        super().__init__(V, Fv, Fx, dim)
        

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def vmax(self) -> float:
        return self._vmax

    def fx(self, x: np.ndarray) -> np.ndarray:
        """
            Model to predict forward
        """
        x_k = np.array([
            x[0] + self.dt * x[2] * np.cos(x[3]),
            x[1] + self.dt * x[2] * np.sin(x[3]), 
            x[2],
            x[3]
        ])
        return x_k
    
    def Fx(self, x : np.ndarray) -> np.ndarray:
        """
            Since these are nonlinear models, the update equations are different.
        """
        fx = np.asarray([#
            [1., 0., self.dt * np.cos(x[3] + self.V[3,3]),  -1. * self.dt * (x[2] + self.V[2,2]) * np.sin(x[3] + self.V[3,3])],
            [0., 1., self.dt * np.sin(x[3], self.V[3,3]),   self.dt * (x[2] * self.V[2,2]) * np.cos(x[3] * self.V[3,3])],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
            ])
        return fx

    def Fv(self, x: np.ndarray) -> np.ndarray:
        fv = np.asarray([#
            [1., 0., self.dt * np.cos(x[3] + self.V[3,3]),  -1. * self.dt * (x[2] + self.V[2,2]) * np.sin(x[3] + self.V[3,3])],
            [0., 1., self.dt * np.sin(x[3], self.V[3,3]),   self.dt * (x[2] * self.V[2,2]) * np.cos(x[3] * self.V[3,3])],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
            ])
        return fv


