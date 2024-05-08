from roboticstoolbox import EKF
from roboticstoolbox.mobile import VehicleBase, VehiclePolygon
from roboticstoolbox.mobile.landmarkmap import LandmarkMap
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation

from mrekf.sensor import SimulationSensor
from mrekf.ekf_base import GT_LOG, BasicEKF
from mrekf.dynamic_ekf import Dynamic_EKF

class Simulation(EKF):
    """ inherited class for the multi-robot problem.
    Only to run the simulation"""

    # Todo - - maybe not - plotting the robot and animating it? could clean the init here much more - do not even need inheritance of Simulation, right?! 
    def __init__(self, robot : VehicleBase, r2 : dict, sensor : SimulationSensor =None, map : LandmarkMap =None, P0=None, x_est=None, joseph=True, animate=True, x0 : np.ndarray=[0., 0., 0.], verbose=False, history=True, workspace=None,
                ekfs : list[EKF] = None
                ):
        super().__init__(robot, sensor=sensor, map=map, P0=P0, x_est=x_est, joseph=joseph, animate=animate, x0=x0, verbose=verbose, history=history, workspace=workspace)
        # Calling arguments:
        # robot=(robot, V), P0=P0, sensor=(sensor, W) ! sensor is now a sensor that also detects the other robot
        if not isinstance(r2, dict):
            raise TypeError("r2 must be a dictionary of robot_id : tuple (vehicle, V_est)")
        
        if not isinstance(ekfs, list):
            raise TypeError("ekfs must be a list of instances of superclass EKF - which have predict and update steps implemented")

        # dict of secondary robots
        self._robots = r2
        
        # Logging Ground Truth
        if history:
            self._keep_history = True
            self._htuple = GT_LOG

        # all EKFs as baselines
        self._ekfs = ekfs

    def __str__(self):
        s = f"Simulation object: {len(self.ekfs)} filters"
        s += "\n {}".format(self.sensor)
        return s

    def __repr__(self):
        return str(self)

    @property
    def robots(self) -> dict:
        return self._robots

    @property
    def ekfs(self) -> list[BasicEKF]:
        return self._ekfs

    ##### Core animation functions:
    def run_animation(self, T=10, x0=None, control=None, format=None, file=None):
        r"""
        Run the EKF simulation

        :param T: maximum simulation time in seconds
        :type T: float
        :param format: Output format
        :type format: str, optional
        :param file: File name
        :type file: str, optional
        :return: Matplotlib animation object
        :rtype: :meth:`matplotlib.animation.FuncAnimation`

        Simulates the motion of a vehicle (under the control of a driving agent)
        and the EKF estimator for ``T`` seconds and returns an animation
        in various formats::

            ``format``    ``file``   description
            ============  =========  ============================
            ``"html"``    str, None  return HTML5 video
            ``"jshtml"``  str, None  return JS+HTML video
            ``"gif"``     str        return animated GIF
            ``"mp4"``     str        return MP4/H264 video
            ``None``                 return a ``FuncAnimation`` object

        If ``file`` can be ``None`` then return the video as a string, otherwise it
        must be a filename.

        The simulation steps are:

        - initialize the filter, vehicle and vehicle driver agent, sensor
        - for each time step:

            - step the vehicle and its driver agent, obtain odometry
            - take a sensor reading
            - execute the EKF
            - save information as a namedtuple to the history list for later display

        :seealso: :meth:`history` :meth:`landmark` :meth:`landmarks`
            :meth:`get_xyt` :meth:`get_t` :meth:`get_map` :meth:`get_P` :meth:`get_Pnorm`
            :meth:`plot_xy` :meth:`plot_ellipse` :meth:`plot_error` :meth:`plot_map`
            :meth:`run_animation`
        """

        fig, ax = plt.subplots()

        def init():
            self.init()
            r_polys = {}
            for r_id, r in self.robots.items():
                r_poly = VehiclePolygon(scale=0.5, color="red")
                r_poly.add(ax)
                r_polys[r_id] = r_poly
            self.__r_polys = r_polys
            if self.sensor is not None:
                self.sensor.map.plot()
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        def animate(i):
            self.robot._animation.update(self.robot.x)
            for r_id, r_poly in self.__r_polys.items():
                r= self.robots[r_id]
                r_poly.update(r.x)
            # for j, r in enumerate(self.robots):
            #     self.__r_polys[j].update(r.x)
            self.step(pause=False)

        nframes = round(T / self.robot._dt)
        anim = animation.FuncAnimation(
            fig=fig,
            func=animate,
            init_func=init,
            frames=nframes,
            interval=self.robot.dt * 1000,
            blit=False,
            repeat=False,
        )

        ret = None
        if format == "html":
            ret = anim.to_html5_video()  # convert to embeddable HTML5 animation
        elif format == "jshtml":
            ret = anim.to_jshtml()  # convert to embeddable Javascript/HTML animation
        elif format == "gif":
            anim.save(
                file, writer=animation.PillowWriter(fps=1 / self.robot.dt)
            )  # convert to GIF
            ret = None
        elif format == "mp4":
            anim.save(
                file, writer=animation.FFMpegWriter(fps=1 / self.robot.dt)
            )  # convert to mp4/H264
            ret = None
        elif format == None:
            # return the anim object
            return anim
        else:
            raise ValueError("unknown format")

        if ret is not None and file is not None:
            with open(file, "w") as f:
                f.write(ret)
            ret = None
        plt.close(fig)
        return ret

    def run_simulation(self, T : int=10) -> None:
        """
            Function to run a simulation without animation
        """
        self.init()
        nsteps = round(T / self.robot._dt)
        for i in tqdm(range(nsteps), desc="Simulation steps"):
            self.step(pause=False)
        return None
    
    
    def get_velocity(self, rob : VehicleBase) -> float:
        """
            Function to get the velocity from a robot
        """
        if len(rob.x_hist) > 1:
            xp = rob.x_hist[-2,0:2]
        else:
            xp = rob.x0[:2]
        xd = rob.x[:2] - xp
        v = np.linalg.norm(xd)
        return v / rob.dt


    def get_velocity(self, rob : VehicleBase) -> float:
        """
            Function to get the velocity from a robot
        """
        if len(rob.x_hist) > 1:
            xp = rob.x_hist[-2,0:2]
        else:
            xp = rob.x0[:2]
        xd = rob.x[:2] - xp
        v = np.linalg.norm(xd)
        return v / rob.dt

    def step(self, pause=None):
        """
            Execute one timestep of the simulation
            # original in file [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/EKF.py]]
            Line 773 
        """
        rsd = {}
        t = self.robot._t
        # move the robot
        odo = self.robot.step(pause=pause)
        for r_id, rob in self.robots.items():
            # ! check function from PC - [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/Vehicle.py]] L 643
            od = rob.step(pause=pause)
            x = rob.x.copy()
            v = self.get_velocity(rob)
            rsd[r_id] = (x, v)
        
        zk, rk = self.sensor.reading()
        z = {**zk, **rk}

        for ekf in self.ekfs:
            ekf.step(t, odo, z, rsd)
            
        if self._keep_history:
            hist = self._htuple(
                self.robot._t,
                self.robot.x.copy(),
                odo.copy(),
                z.copy(),
                rsd
                )
            self._history.append(hist)