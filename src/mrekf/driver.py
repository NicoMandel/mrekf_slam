import numpy as np
from roboticstoolbox.mobile import RandomPath

class DynamicPath(RandomPath):
    """
        Class that changes the velocity depending on the steering angle and accelerating.
        VehicleBase.step ((roboticstoolbox/mobile/Vehicle.py/Vehiclebase)):
        u=self.eval_control(self._control, self._x)
        xd = self._dt * self.deriv(self._x, u)
        -> Bicycle.deriv: v, gamma = u. return v * np.r_[cos(theta), sin(theta), tan(gamma) / self.l]
        self._x += xd
        eval_control(self, control, x) does:
        if isinstance(control, VehicleDriverBase) : u = control.demand()
    """

    def __init__(self, workspace, speed=1, dthresh=0.05, seed=0, headinggain=0.3, goalmarkerstyle=None):
        super().__init__(workspace, speed, dthresh, seed, headinggain, goalmarkerstyle)

        # self._heading_limit = np.deg2rad(heading_lim)
        # self._u_list = []

    @property
    def heading_limit(self):
        return self._heading_limit
    
    def demand(self) -> np.ndarray:
        """
            ! Overwrite "demand"
            u[1] is in radians
        """
        u : np.ndarray = super().demand()
        # speedgain = (self.heading_limit - np.abs(u[1])) / self.heading_limit
        speedgain = np.cos(np.abs(u[1]) / (1. - self._headinggain))
        # print("Original speed: {:.4f}. New speed: {:.4f}".format(u[0], speedgain * u[0]))
        v = speedgain * u[0]
        u[0] = 0.1 if v < 0.1 else v
        # self._u_list.append(u)
        return u