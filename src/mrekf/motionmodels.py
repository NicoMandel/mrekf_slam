
import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):

    # Motion Models
    @abstractmethod
    def f(self, x):
        pass
    
    @abstractmethod
    def Fv(self, x : np.ndarray = None) -> np.ndarray:
        pass

    @abstractmethod
    def Fx(self, x : np.ndarray = None) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def state_length(self) -> int:
        pass

    @property
    @abstractmethod
    def V(self) -> np.ndarray:
        pass

    @abstractmethod
    def scale_V(self, scale : float = None) -> np.ndarray:
        pass

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
        self._V = V
        dim = 2
        self._state_length = dim
                
        # derivative matrices
        self._Fv = np.eye(dim, dtype=float)
        self._Fx = np.eye(dim, dtype=float)
    
    @property
    def V(self):
        return self._V
    
    @property
    def state_length(self):
        return self._state_length

    def f(self, x : np.ndarray = None) -> np.ndarray:
        """
            f(x_k+1) = x_k + v_x
        """
        return x

    def Fx(self, x : np.ndarray = None) -> np.ndarray:
        return self._Fx
    
    def Fv(self, x : np.ndarray = None) -> np.ndarray:
        return self._Fv
    
    def scale_V(self, scale : float = None) -> np.ndarray:
        return scale * self.V if scale is not None else self.V
    

class KinematicModel(BaseModel):
    """
        Class to provide a kinematic motion model
        defined as 
        :math:  x_k+1 = x_k + dt * (x_k_dot + v_x)
        :math:  y_k+1 = y_k + dt * (y_k_dot + v_y)
        :math:  y_k_dot+1 = y_k_dot + v_y
        :math:  x_k_dot+1 = x_k_dot + v_x
    """

    def __init__(self, V : np.ndarray, dt : float) -> None:
        assert V.shape == (4,4), "V not correct shape, Please make sure it's 4x4"
        self._V = V
        self._state_length = 4
        self._dt = dt

        # A matrix for state prediction
        self._A = np.array([
            [1., 0., dt, 0.],
            [0., 1., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])

        # B Matrix for Noise
        self._B = np.array([
            [0., 0., dt, 0.],
            [0., 0., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
        
        # derivative Matrices
        self._Fx = np.array([
            [1., 0., dt, 0.],
            [0., 1., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])

        self._Fv = np.array([
            [0., 0., dt, 0.],
            [0., 0., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])

    @property
    def V(self) -> np.ndarray:
        return self._V
    
    @property
    def state_length(self) -> int:
        return self._state_length
    
    @property
    def dt(self) -> float:
        return self._dt

    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def B(self) -> np.ndarray:
        return self._B

    def f(self, x : np.ndarray = None) -> np.ndarray:
        fx_k = self.A @ x
        return fx_k

    def Fx(self, x : np.ndarray = None) -> np.ndarray:
        return self._Fx
    
    def Fv(self, x : np.ndarray = None) -> np.ndarray:
        return self._Fv
    
    def scale_V(self, scale : float = None) -> np.ndarray:
        return scale * self.V if scale is not None else self.V
        
    
