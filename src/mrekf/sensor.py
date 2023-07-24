import numpy as np
from spatialmath import base
from roboticstoolbox import RangeBearingSensor
from roboticstoolbox.mobile import VehicleBase, LandmarkMap
from mrekf.motionmodels import KinematicModel, StaticModel, BaseModel

class RobotSensor(RangeBearingSensor):
    """
        Inherit a Range and Bearing Sensor
        senses map and other robots
    """

    def __init__(self, robot : VehicleBase, r2 : list, lm_map : LandmarkMap, line_style=None, poly_style=None, covar=None, range=None, angle=None, plot=False, seed=0, **kwargs):
        if not isinstance(r2, list):
            raise TypeError("Robots should be a list of other robots. Of the specified robot format")
        self._r2s = r2
        super().__init__(robot, map=lm_map, line_style=line_style, poly_style=poly_style, covar=covar, range=range, angle=angle, plot=plot, seed=seed, **kwargs)
        
    # using function .h(x, p) is range and bearing to landmark with coordiantes p
    # .reading() adds noise - multivariate normal with _W in it
    # function .reading() from sensor only takes ONE landmark!
    # see [[/home/mandel/mambaforge/envs/corks/lib/site-packages/roboticstoolbox/mobile/sensors.py]]
    # line 433
    
    @property
    def r2s(self):
        return self._r2s

    def visible_rs(self):
        """
            Function to return a visibility reading on the robots in the vicinity.
            Sames as for lms. Lms are inherited
        """
        zk = []
        for i, r in enumerate(self.r2s):
            z = self.h(self.robot.x, (r.x[0], r.x[1])) # measurement function
            zk.append((z, i))
            # zk = [(z, k) for k, z in enumerate(z)]
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
        zk = self.visible()
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

    def __init__(self, robot: VehicleBase, r2: list, lm_map: LandmarkMap, line_style=None, poly_style=None, covar=None, range=None, angle=None, plot=False, seed=0, **kwargs):
        super().__init__(robot, r2, lm_map, line_style, poly_style, covar, range, angle, plot, seed, **kwargs)

    # todo continue here - overwrite h, Hx, Hp and Hw (maybe use parent functions and just append)
    def h(self, x : np.ndarray, landmark = None):
        """
            x is always the robot state
        """
        if landmark is not None and isinstance(landmark, np.ndarray) and landmark.shape[0] == 4:        # condition when to use the kinematic sensing function
            return self._h(x, landmark)
        else:
            return super().h(x, landmark)
    
    def _h(self, x: np.ndarray, r_state : np.ndarray = None):
        """
            hidden specific function only for the robot state 
            defined in L 555 ctd of:
            [[/home/mandel/mambaforge/envs/corke/lib/python3.1/site-packages/roboticstoolbox/mobile/sensors.py]]
        """
        x, y, t = x
        assert isinstance(r_state, np.ndarray), "robot state is not a numpy array! Check again"
        
        # landmark quadruplet of position and velocities
        xlm = base.getvector(r_state, 4)
        dx = xlm[0] - x
        dy = xlm[1] - y

        # turn into an actual measurement
        z = np.c_[
            np.sqrt(dx**2 + dy**2), base.angdiff(np.arctan2(dy, dx), t)
        ]  # range & bearing as columns
        return z

    

def get_sensor_model(motion_model : BaseModel, covar : np.ndarray, robot : VehicleBase, r2 : list, lm_map : LandmarkMap, rng, **kwargs) -> RobotSensor:
    """
        helper function to get the right 
    """
    if isinstance(motion_model, KinematicModel):
        return KinematicSensor(
            robot=robot,
            r2 = r2,
            lm_map = lm_map,
            range=rng,
            covar = covar,
            **kwargs
            )
    else:
        return RobotSensor(
            robot=robot,
            r2=r2,
            lm_map=lm_map,
            covar=covar,
            range=rng,
            **kwargs            
        )