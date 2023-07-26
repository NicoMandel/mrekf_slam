import numpy as np
from spatialmath import base
from roboticstoolbox import RangeBearingSensor
from roboticstoolbox.mobile import VehicleBase, LandmarkMap
from mrekf.motionmodels import KinematicModel, StaticModel, BaseModel

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

    def _is_kinematic(self, landmark) -> bool:
        """
            helper function to figure out if a landmark is kinematic or not
        """
        return True if (landmark is not None and isinstance(landmark, np.ndarray) and landmark.shape[0] == 4) else False


    # todo continue here - overwrite h and Hp (Hw and Hx are unchanged) (maybe use parent functions and just append)
    def h(self, x : np.ndarray, landmark = None):
        """
            x is always the robot state
        """
        if self._is_kinematic(landmark):        # condition when to use the kinematic sensing function
            lm_v = landmark[2:]
            landmark = landmark[:2]
            is_kin = True
        else:
            is_kin = False
        out =  super().h(x, landmark)
        if is_kin:
            out = np.r_[
                    out,
                    np.zeros((2,2))
                    ]
        return out

    # Hx and Hw are unchanged! Hp changes    
    def Hp(self, x, landmark):
        out = super().Hp(x, landmark)
        if self._is_kinematic(landmark):
            out = np.c_(out, np.zeros((2,2)))
        return out
    
    # Insertion functions g, Gx, Gz
    def g(self, x : np.ndarray, z : np.ndarray, is_kinematic : bool = False):
        """
            Insertion function g. Requires a flag whether the own insertion function will be called
            Todo: figure out where to get v_max from. should be part of the motion model.
        """
        g_f =  super().g(x, z)
        if is_kinematic:
            g_f = np.r_[g_f,
                        v_max,
                        v_max
                        ]
        return g_f
    
    def Gx(self, x : np.ndarray, z : np.ndarray, is_kinematic : bool = False):
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
    
    def Gz(self, x: np.ndarray, z : np.ndarray, is_kinematic : bool = False):
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

    # insertion functions y and Yz

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