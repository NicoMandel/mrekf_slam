
import numpy as np
from roboticstoolbox.mobile import VehicleBase
from abc import ABC, abstractmethod, abstractproperty

class BaseModel(ABC):

    # Motion Models
    @abstractmethod
    def f(self, x):
        pass
    
    @abstractmethod
    def Fv(self, x : np.ndarray = None):
        pass

    @abstractmethod
    def Fx(self, x : np.ndarray = None):
        pass

    @property
    @abstractmethod
    def state_length(self) -> int:
        pass

    @property
    @abstractmethod
    def V(self) -> np.ndarray:
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
        self._state_length = 2
    
    @property
    def V(self):
        return self._V
    
    @property
    def state_length(self):
        return self._state_length

    def f(self, x : np.ndarray) -> np.ndarray:
        """
            f(x_k+1) = x_k + v_x
        """
        return x

    def Fx(self, x : np.ndarray = None) -> np.ndarray:
        dim = self._state_length
        fx = np.eye(dim, dim)
        return fx
    
    def Fv(self, x : np.ndarray = None) -> np.ndarray:
        dim = self._state_length
        fv = np.eye(dim,dim)
        return fv
    

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
        pass
