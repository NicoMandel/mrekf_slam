import numpy as np
from spatialmath import base
from roboticstoolbox import RangeBearingSensor
from roboticstoolbox.mobile import VehicleBase, LandmarkMap
from mrekf.motionmodels import KinematicModel, BaseModel, BodyFrame

"""
    Todo:
    how to inialise velocities?
        a) only insert on second observation, use observed v
        b) insert with the highest assumed value, so that it decrease?
        c) ensure it never goes to 0?
"""

class RobotSensor(RangeBearingSensor):
    """
        Inherit a Range and Bearing Sensor
        senses map and other robots
    """

    def __init__(self, robot : VehicleBase, r2 : dict, lm_map : LandmarkMap, robot_offset : int = 100, line_style=None, poly_style=None, covar=None, range=None, angle=None, plot=False, seed=0, **kwargs):
        """
            :param robot: model of robot carrying the sensor
        """
        
        if not isinstance(r2, dict):
            raise TypeError("Robots should be a dict of robots with id : tuple(robot, v_est). Of the specified robot format")
        self._r2s = r2
        self._robot_offset = robot_offset
        super().__init__(robot, map=lm_map, line_style=line_style, poly_style=poly_style, covar=covar, range=range, angle=angle, plot=plot, seed=seed, **kwargs)
        
    # using function .h(x, p) is range and bearing to landmark with coordiantes p
    # .reading() adds noise - multivariate normal with _W in it
    # function .reading() from sensor only takes ONE landmark!
    # see [[/home/mandel/mambaforge/envs/corks/lib/site-packages/roboticstoolbox/mobile/sensors.py]]
    # line 433

    # overwriting sensor functions to include the is_kinematic bool - but actually just ignore it.
    def Hp(self, x, landmark, is_kinematic : bool = False) -> np.ndarray:
        return super().Hp(x, landmark)

    def g(self, x : np.ndarray, z : np.ndarray, is_kinematic : bool = False) -> np.ndarray :
        return super().g(x, z)

    def Gx(self, x : np.ndarray, z : np.ndarray, is_kinematic : bool = False)  -> np.ndarray:
        return super().Gx(x, z)

    def Gz(self, x: np.ndarray, z : np.ndarray, is_kinematic : bool = False) -> np.ndarray:
        return super().Gz(x, z)
    
    @property
    def r2s(self) -> dict:
        return self._r2s

    @property
    def robot_offset(self) -> int:
        return self._robot_offset

    # overwriting the .visible() functions - to ensure the filter functions are working as desired - do not give right values simulation
    def _within_r(self, val : tuple) -> bool:
        """
            Function to check whether a distance value is within the sensor thresholds
            :param val: observation ((r, theta), id)
            :type val: tuple
            :return: whether a value is within the distance or not
            :rtype: bool
        """
        return True if self._r_range[0] <= val[0][0] <= self._r_range[1] else False
    
    def _within_theta(self, val : tuple) -> bool:
        """
            Function to check whether an angular value is within the sensor thresholds
            :param val:  observation ((r, theta), id)
            :type val: tuple
            :return: whether a value is within the angle or not
            :rtype: bool
        """
        return True if self._theta_range[0] <= val[0][1] <= self._theta_range[1] else False
    
    
    def visible_lms(self) -> list:
        """
            Overwritten visibility function from original implementation
            -> the filter function did not work as desired and threw weird errors. Therefore we will make the filter functions explicit
            :return: list of all visible lms (static)
            :rtype: list
        """
        z = super().h(self.robot.x)
        zk = [(z, k) for k, z in enumerate(z)]

        if self._r_range is not None:
            # zk = filter(self._within_r, zk)
            zk = [zi for zi in zk if self._within_r(zi)]

        if self._theta_range is not None:
            # zk = filter(self._within_theta, zk)
            zk = [zi for zi in zk if self._within_theta(zi)]
        
        return list(zk)

    def visible_rs(self) -> list:
        """
            Function to return a visibility reading on the robots in the vicinity.
            Sames as for lms. Lms are inherited
            :return: list of all visible robots
            :rtype: list
        """
        zk = []
        for r_id, r in self.r2s.items():        
            z = self.h(self.robot.x, (r.x[0], r.x[1])) # measurement function
            zk.append((z, r_id))
        # filter by range
        if self._r_range is not None:
            zk = filter(lambda zk: self._r_range[0] <= zk[0][0] <= self._r_range[1], zk)

        if self._theta_range is not None:
            # find all within angular range as well
            zk = filter(
            lambda zk: self._theta_range[0] <= zk[0][1] <= self._theta_range[1], zk
        )

        return list(zk)
            
    def reading(self):
        """
            Function to return a reading of every visible landmark. Same format as PC
            noise with covariance W is added to the reading
            function to return landmarks and visible robots
        """
        # prelims - same as before
        self._count += 1

        # list of visible landmarks
        zk = self.visible_lms()
        rs = self.visible_rs()
        # ids = zk[:][0] 
        # meas = zk[:][1] 
        # add multivariate noise
        zzk = {idd : m + self._random.multivariate_normal((0,0), self._W)  for m, idd in zk}
        rrk = {idd : m + self._random.multivariate_normal((0,0), self._W)  for m, idd in rs}

        return zzk, rrk
       

class KinematicSensor(RobotSensor):
    """
        Class to provide the functions and methods if the state model of the robot is 4 kinematic states
    """

    def __init__(self, robot: VehicleBase, r2: list, lm_map: LandmarkMap, motion_model : KinematicModel, robot_offset : int = 100, line_style=None, poly_style=None, covar=None, range=None, angle=None, plot=False, seed=0, **kwargs):
        
        self._kin_model = motion_model
        super().__init__(robot, r2, lm_map=lm_map, robot_offset=robot_offset, line_style=line_style, poly_style=poly_style, covar=covar, range=range, angle=angle, plot=plot, seed=seed, **kwargs)
      

    # overwrite only Hp (h, Hw and Hx are unchanged)
    # Hx and Hw are unchanged! Hp changes    
    def Hp(self, x, landmark, is_kinematic : bool = False) -> np.ndarray:
        out = super().Hp(x, landmark)
        if is_kinematic:
            out = np.c_[out, np.zeros((2,2))]
        return out
    
    # Insertion functions g, Gx, Gz
    def g(self, x : np.ndarray, z : np.ndarray, is_kinematic : bool = False) -> np.ndarray :
        """
            Insertion function g. Requires a flag whether the own insertion function will be called
        """
        g_f =  super().g(x, z)
        if is_kinematic:
            v_max = self._kin_model.vmax
            g_f = np.r_[g_f,
                        v_max,
                        v_max
                        ]
        return g_f
    
    def Gx(self, x : np.ndarray, z : np.ndarray, is_kinematic : bool = False)  -> np.ndarray:
        """
            Jacobian dg / dx
        """
        G_x = super().Gx(x, z)
        if is_kinematic:
            G_x = np.r_[
                G_x,
                np.zeros((2,3))
            ]
        return G_x
    
    def Gz(self, x: np.ndarray, z : np.ndarray, is_kinematic : bool = False) -> np.ndarray:
        """
            Jacobian dg / dz
        """
        G_z = super().Gz(x, z)
        if is_kinematic:
            G_z = np.r_[
                G_z,
                np.zeros((2,2))
            ]
        return G_z        


def get_sensor_model(motion_model : BaseModel, covar : np.ndarray, robot : VehicleBase, r2 : dict, lm_map : LandmarkMap, rg, robot_offset : int = 100, **kwargs) -> RobotSensor:
    """
        helper function to get the right 
    """
    if isinstance(motion_model, KinematicModel) or isinstance(motion_model, BodyFrame):
        return KinematicSensor(
            robot=robot,
            r2 = r2,
            lm_map = lm_map,
            robot_offset=robot_offset,
            range=rg,
            covar = covar,
            motion_model = motion_model,
            **kwargs
            )
    else:
        return RobotSensor(
            robot=robot,
            r2=r2,
            lm_map=lm_map,
            robot_offset=robot_offset,
            range=rg,
            covar=covar,
            **kwargs            
        )