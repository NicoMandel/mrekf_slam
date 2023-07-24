
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

    def __init__(self, V : np.ndarray, dt : float) -> None:
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
    def A(self) -> np.ndarray:
        return self._A


    def f(self, x : np.ndarray) -> np.ndarray:
        fx_k = self.A @ x
        return fx_k   
