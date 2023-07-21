
from roboticstoolbox.mobile import VehicleBase

class StaticModel:
    """
        Class to provide a static motion model.
        defined as 
        :math:  x_k+1 = x_k + v_x
        :math:  y_k+1 = y_k + v_y
    """

    def __init__(self) -> None:
        pass

class KinematicModel:
    """
        Class to provide a kinematic motion model
        defined as 
        :math:  x_k+1 = x_k + dt * (x_k_dot + v_x)
        :math:  y_k+1 = y_k + dt * (y_k_dot + v_y)
        :math:  y_k_dot+1 = y_k_dot + v_y
        :math:  x_k_dot+1 = x_k_dot + v_x
    """

    def __init__(self) -> None:
        pass
