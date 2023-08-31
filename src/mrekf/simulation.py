from typing import Tuple
from collections import namedtuple

from roboticstoolbox import EKF, RangeBearingSensor
from roboticstoolbox.mobile import VehicleBase, VehiclePolygon
from roboticstoolbox.mobile.landmarkmap import LandmarkMap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from mrekf.sensor import RobotSensor
from mrekf.ekf_base import GT_LOG
from mrekf.ekf_math import *

class Simulation(EKF):
    """ inherited class for the multi-robot problem.
    Only to run the simulation"""

    def __init__(self, robot, r2 : list, sensor : RobotSensor =None, map=None, P0=None, x_est=None, joseph=True, animate=True, x0 : np.ndarray=[0., 0., 0.], verbose=False, history=True, workspace=None,
                ekfs : list[EKF] = None 
                ):
        super().__init__(robot, sensor=sensor, map=map, P0=P0, x_est=x_est, joseph=joseph, animate=animate, x0=x0, verbose=verbose, history=history, workspace=workspace)
        # Calling arguments:
        # robot=(robot, V), P0=P0, sensor=(sensor, W) ! sensor is now a sensor that also detects the other robot
        if not isinstance(r2, list):
            raise TypeError("r2 must be a list. Must also be tuple (vehicle, V_est)")
        
        if not isinstance(ekfs, list):
            raise TypeError("ekfs must be a list of instances of superclass EKF - which have predict and update steps implemented")

        # list of robots. and needs list of seen robots
        self._robots = r2
        
        # Logging Ground Truth
        self._htuple = GT_LOG

        # extra EKFs as baselines
        self._ekfs = ekfs

    def __str__(self):
        s = super().__str__()
        def indent(s, n=2):
            spaces = " " * n
            return s.replace("\n", "\n" + spaces)
        
        if self.robots:
            s += indent("\nEstimating {} robots:  ".format(self.robots))

        return s

    def __repr__(self):
        return str(self)

    @property
    def robots(self):
        return self._robots

    @property
    def ekfs(self):
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
            r_polys = []
            for r in self.robots:
                r_poly = VehiclePolygon(scale=0.5, color="red")
                r_poly.add()
                r_polys.append(r_poly)
            self.__r_polys = r_polys
            if self.sensor is not None:
                self.sensor.map.plot()
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        def animate(i):
            self.robot._animation.update(self.robot.x)
            for j, r in enumerate(self.robots):
                self.__r_polys[j].update(r.x)
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


    def step(self, pause=None):
        """
            Execute one timestep of the simulation
            # original in file [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/EKF.py]]
            Line 773 
        """
        rsd = {}
        # move the robot
        odo = self.robot.step(pause=pause)
        for j, rob in enumerate(self.robots):
            # todo - find a way to get correspondence here between robot index in the sim and robot index in the sensor
            # turn r2s into a dictionary -> with id as id and rest as data inside -> get the ide through dict index?!
            # should be straightforward to fix in the sensor
            # ! check function from PC - [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/Vehicle.py]] L 643
            od = rob.step(pause=pause)
            rsd[j + self.sensor.robot_offset] = rob.x.copy()
        
        zk, rk = self.sensor.reading()
        z = zk + rk
        t = self.robot._t
        for ekf in self.ekfs:
            ekf.step(t, odo, z)
            
        if self._keep_history:
            hist = self._htuple(
                self.robot._t,
                self.robot.x.copy(),
                odo.copy(),
                z.copy(),
                rsd
                )
            self._history.append(hist)