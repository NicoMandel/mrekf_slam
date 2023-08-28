from typing import Tuple
from collections import namedtuple

from roboticstoolbox import EKF, RangeBearingSensor
from roboticstoolbox.mobile import VehicleBase, VehiclePolygon
from roboticstoolbox.mobile.landmarkmap import LandmarkMap
import numpy as np
from spatialmath import base
import matplotlib.pyplot as plt
from matplotlib import animation

from mrekf.sensor import RobotSensor
from mrekf.ekf_base import EKF_base, MR_EKFLOG
from mrekf.motionmodels import BaseModel, KinematicModel, BodyFrame
from mrekf.ekf_math import *

""" 
    TODO topics:
        1. include a counter for the instances of robots. so that we know which robot and how many. and then a mapping that corresponds from true robot id to estimated robot in the maps 
        4. could theoretically use get_transformation_params() to calculate the TE at certain timesteps (instead of as average over time) and show the impact of re-including the lm observation
        5. If we want to show the positive impact of using dynamic landmarks, we can show that the scale diverges majorly through false negatives
            for that we need another metric - one that does not rotate and rescale the map, so we would use the absolute distance from the true track
        6. write a function which plots the aligned maps, with the parameters from the calculate_map_alignment
        7. investigate the relative increase in time when the robot is seen vs. when it is not seen
        8. Make the icon transparent when the robot is not observed. Needs 1.
    ! Make sure the update step only happens if there is actually information in the innovation!
    ! double check both usages of self.get_W_est() - are they the same length that are inserted?!
    otherwise it shouldn't happen at all!!!
    Displaying the second robot is really difficult - there is some issue on setting the xdata and plotting. it is never showed in the second plot
    see Animations file liens 155 - 162 - updating and plotting!
    TOdo:
        * correct plotting
            * ensure that plotting of the second robot is only done when the robot is observed
            * state estimates
        * https://stackoverflow.com/questions/2010692/what-does-mro-do - use MRO to define subclasses of everything
"""


class EKF_MR(EKF):
    """ inherited class for the multi-robot problem"""

    def __init__(self, robot, r2 : list, motion_model : BaseModel, sensor : RobotSensor =None, map=None, P0=None, x_est=None, joseph=True, animate=True, x0 : np.ndarray=[0., 0., 0.], verbose=False, history=True, workspace=None,
                EKF_include : EKF_base = None, EKF_exclude : EKF_base = None, EKF_fp = None
                ):
        super().__init__(robot, sensor=sensor, map=map, P0=P0, x_est=x_est, joseph=joseph, animate=animate, x0=x0, verbose=verbose, history=history, workspace=workspace)
        # Calling arguments:
        # robot=(robot, V), P0=P0, sensor=(sensor, W) ! sensor is now a sensor that also detects the other robot
        if not isinstance(r2, list):
            raise TypeError("r2 must be a list. Must also be tuple (vehicle, V_est)")
        
        # list of robots. and needs list of seen robots
        self._robots = r2
        self._seen_robots = {}

        # Management of the model that the agent has of the other robot
        self._motion_model = motion_model

        # Logging tuples
        self._htuple = MR_EKFLOG

        # extra EKFs as baselines
        self.ekf_include = EKF_include
        self.ekf_exclude = EKF_exclude
        self.ekf_FP = EKF_fp

    def __str__(self):
        s = super(EKF_MR, self).__str__()
        def indent(s, n=2):
            spaces = " " * n
            return s.replace("\n", "\n" + spaces)
        
        if self.robots:
            s += indent("\nEstimating {} robots:  ".format(self.robots))
            s+=indent("\nV of others:  " + base.array2str(self._V_model))

        return s

    def __repr__(self):
        return str(self)

    @property
    def robots(self):
        return self._robots
    
    @property
    def seen_robots(self):
        return self._seen_robots

    @property
    def motion_model(self) -> BaseModel:
        return self._motion_model

    def has_kinematic_model(self) -> bool:
        return True if isinstance(self.motion_model, KinematicModel) or isinstance(self.motion_model, BodyFrame) else False


    ######## Section on Landmark and Robot Management
    def get_state_length(self):
        """
            Function to get the current length of the state vector - to figure out where to append
        """
        return 3 + self.motion_model.state_length * len(self._seen_robots) + 2 * len(self.landmarks)

    # Robot Management
    def _isseenbefore_robot(self, r_id):
        # robots[id], 0 is the order in which seen
        # robots[id], 1 is the occurence count
        # robots[id], 2 is the index in the map

        return r_id in self._seen_robots

    def _robot_increment(self, r_id):
        self._seen_robots[r_id][1] += 1  # update the count

    def _robot_count(self, r_id):
        return self._seen_robots[r_id][1]

    def _robot_add(self, r_id):
        """
            Updated function -> inserting into a new position.
        """
        pos = self.get_state_length()
        self._seen_robots[r_id] = [len(self._seen_robots), 1, pos]

    def get_robot_info(self, r_id):
        """
            Robot information.
            first index is the order in which seen.
            second index is the count
            third index is added - it's the map index
        """
        try:
            r = self._seen_robots[r_id]
            return r[0], r[1], r[2]
        except KeyError:
            raise ValueError("Unknown Robot ID: ".format(r_id))

    def robot_index(self, r_id):
        """
            index in the complete state vector
        """
        try:
            return self._seen_robots[r_id][2]
        except KeyError:
            raise ValueError(f"unknown robot {r_id}") from None
    
    def robot_mindex(self, r_id):
        return self.robot_index(r_id)
    
    def get_xyt(self):
        r"""
        straight from PC - for inheritance issues
        Get estimated vehicle trajectory

        :return: vehicle trajectory where each row is configuration :math:`(x, y, \theta)`
        :rtype: ndarray(n,3)

        :seealso: :meth:`plot_xy` :meth:`run` :meth:`history`
        """
        
        xyt = np.array([h.xest[:3] for h in self._history])
        return xyt


    ### Overwriting the landmark functions for consistency -> in our case only want the position in the full vector dgaf about map vector
    def _landmark_add(self, lm_id):
        pos = self.get_state_length()
        self._landmarks[lm_id] = [len(self._landmarks), 1, pos]

    def landmark_index(self, lm_id):
        try:
            jx = self._landmarks[lm_id][2]
            return jx
        except KeyError:
            raise ValueError("Unknown landmark: {}".format(lm_id))

    def landmark_mindex(self, lm_id):
        return self.landmark_index(lm_id)
    
    def landmark_x(self, id):
        """
        straight from PC. to prevent inheritance issues
        Landmark position

        :param id: landmark index
        :type id: int
        :return: landmark position :math:`(x,y)`
        :rtype: ndarray(2)

        Returns the landmark position from the current state vector.
        """
        jx = self.landmark_index(id)
        return self._x_est[jx : jx + 2]

    ###### helper functions for getting the right matrices
    def predict_robots(self, x_pred : np.ndarray) -> np.ndarray:
        """
            applying the respective prediction functions for the robots
        """
        mmsl = self.motion_model.state_length
        for r in self.seen_robots:
            r_ind = self.robot_index(r)
            x_r = x_pred[r_ind : r_ind + mmsl]
            x_e = self.motion_model.f(x_r)
            x_pred[r_ind : r_ind + mmsl] = x_e
        return x_pred

    def get_Fx(self, odo) -> np.ndarray:
        """
            Function to create the full Fx matrix.#
            Fx of the robot is self.robot.Fx(odo)
            Fx of static landmarks is np.zeros()
            Fx of dynamic landmarks is np.eye(), which is in self.F
        """
        # create a large 0 - matrix
        dim = len(self._x_est)
        Fx = np.zeros((dim, dim))
        # insert the robot Jacobian
        xv_est = self.x_est[:3]
        v_Fx = self.robot.Fx(xv_est, odo)
        Fx[:3,:3] = v_Fx
        # insert the jacobian for all dynamic landmarks
        mmsl = self.motion_model.state_length
        for r in self.seen_robots or []:      # careful - robots is not seen robots!
            r_ind = self.robot_index(r)
            xr = self.x_est[r_ind : r_ind + mmsl]
            Fx[r_ind : r_ind + mmsl, r_ind : r_ind + mmsl] = self.motion_model.Fx(xr)

        return Fx

    def get_Fv(self, odo) -> np.ndarray:
        """
            Function to create the full Fv matrix
        """
        # create a large 0 - matrix - one less column, because v is 2, but xv is 3
        dim = len(self.x_est)
        Fv = np.zeros((dim, dim -1))
        # insert the robot Jacobian
        xv_est = self.x_est[:3]
        v_Fv = self.robot.Fv(xv_est, odo)
        Fv[:3,:2] = v_Fv
        # insert the jacobian for all dynamic landmarks
        mmsl = self.motion_model.state_length
        for r in self.seen_robots or []:      # careful - robots is not seen robots!
            r_ind = self.robot_index(r)
            xr = self.x_est[r_ind : r_ind + mmsl]
            ins_r = r_ind
            ins_c = r_ind - 1
            Fv[ins_r : ins_r + mmsl, ins_c : ins_c + mmsl] = self.motion_model.Fv(xr)

        return Fv

    def get_V(self) -> np.ndarray:
        """
            Function to get the full V matrix
        """
        # create a large 0 - matrix - is 1 less than the state, because measurements are 2s and robot state is 3
        dim = len(self.x_est) -1
        Vm = np.zeros((dim, dim))
        # insert the robot Noise model - which is roughly accurate
        V_v = self.robot._V
        Vm[:2, :2] = V_v
        # for each dynamic landmark, insert an index 
        mmsl = self.motion_model.state_length
        for r in self.seen_robots or []:
            # is one less, because the state for the robot is 3x3, but the noise model is only 2x2
            r_ind = self.robot_index(r) -1
            Vm[r_ind : r_ind + mmsl, r_ind : r_ind + mmsl] = self.motion_model.V

        return Vm
    
    ######## Sensor Reading section
    def split_readings(self, readings, test_fn):
        """
            function to split into seen and unseen. test_fn for functional programming
        """
        seen = {}
        unseen = {}
        for lm_id, z in readings.items():
            if test_fn(lm_id):
                seen[lm_id] = z
            else:
                unseen[lm_id] = z
        return seen, unseen
    
    def fuse_observations(self, zk : dict, rk : dict):
        """
            Function to fuse the observations
            use an offset for the key of the rks
            > 100 
        """
        zzk = zk.copy()    # return a copy of the array 
        for r_id, z in rk.items():
            zzk[r_id] = z     # change this! - removed +100
        return zzk

    def get_innovation(self, x_pred : np.ndarray, seen_lms, seen_rs) -> np.ndarray:
        """
            Function to calculate the innovation.
            we calculate in such a way that the innovation is 0 if it is not observed - taking advantage of Linear Algebra.
            This way the landmark update doesn't get propagated into the state if we apply K to it. 
            This is important for the update of x_k to be calculated into the right place
            so even in a kinematic state case, we get 0 in the innovation from the velocities!
        """
        # one innovation for each lm
        innov = np.zeros(len(x_pred) - 3)
        # get the predicted vehicle state
        xv_pred = x_pred[:3]
        for lm_id, z in seen_lms.items():
            # get the index of the landmark in the map and the corresponding state
            m_ind = self.landmark_mindex(lm_id)
            xf = x_pred[m_ind : m_ind + 2]
            # z is the measurement, z_pred is what we thought the measurement should be.
            z_pred = self.sensor.h(xv_pred, xf)
            # the lm- specific innnovation
            inn = np.array(
                    [z[0] - z_pred[0], base.wrap_mpi_pi(z[1] - z_pred[1])]
                )
            # have to index 3 less into the innovation - because innovation does not contain the 3 vehicle states
            innov[m_ind -3 : m_ind -1] = inn
            # Update the landmark count:
            self._landmark_increment(lm_id)


        for r_id, z in seen_rs.items():
            # get the index of the landmark in the map and the corresponding state
            m_ind = self.robot_index(r_id)
            xf = x_pred[m_ind : m_ind + 2]
            # z is the measurement, z_pred is what we thought the measurement should be.
            z_pred = self.sensor.h(xv_pred, xf)   
            # the lm- specific innnovation
            inn = np.array(
                    [z[0] - z_pred[0], base.wrap_mpi_pi(z[1] - z_pred[1])]
                )
            # have to index 3 less into the innovation - because the innovation does not contain the 3 vehicle states
            innov[m_ind -3 : m_ind -1] = inn

            # update the robot count
            self._robot_increment(r_id)
            
        return innov

    def get_Hx(self, x_pred : np.ndarray, seen_lms : dict, seen_rs : dict) -> np.ndarray:
        """
            Function to get a large state measurement jacobian
            the jacobian here is not 2x... but (2xzk) x ... . See Fenwick (2.24)
        """
        # set up a large zero-matrix
        dim = x_pred.size 
        Hx = np.zeros((dim -3, dim))    # three less rows. vechicle is not included (-3)
        xv_pred = x_pred[:3]
        # go through all static landmarks
        for lm_id, _ in seen_lms.items():
            # get the landmark index in the map and the state estimate
            l_ind = self.landmark_index(lm_id)
            xf = x_pred[l_ind : l_ind + 2]
            # calculate BOTH jacobians with respect to the current position and estimate and state estimate
            Hp_k = self.sensor.Hp(xv_pred, xf)
            Hxv = self.sensor.Hx(xv_pred, xf)
            # insert the vehicle Jacobian in the first COLUMN - corresponding to the first three states
            # index is 4 before. see dimensionality
            l_mind = l_ind - 3
            Hx[l_mind: l_mind+2, :3] = Hxv
            # landmark index is 1 row before, because Hxv is only 2 rows
            # ? should the columns be l_ind or l_mind?
            Hx[l_mind : l_mind+2, l_ind : l_ind+2] = Hp_k 

        # go through all dynamic landmarks
        is_kinematic = self.has_kinematic_model()
        mmsl = self.motion_model.state_length
        for r_id, _ in seen_rs.items():
            # get robot index
            r_ind = self.robot_index(r_id)
            xf = x_pred[r_ind : r_ind + mmsl]
            xf_pos = xf[:2]
            # calculate BOTH jacobians with respect to the current position and estimate and state estimate
            Hp_k = self.sensor.Hp(xv_pred, xf_pos, is_kinematic)
            Hxv = self.sensor.Hx(xv_pred, xf_pos)
            # insert the vehicle Jacobian in the first COLUMN - corresponding to the first three states
            r_mind = r_ind - 3       # see above
            Hx[r_mind : r_mind+2, :3] = Hxv
            # robot index is 1 row before, because Hxv is only 2 rows
            # ? should the columns be r_ind or r_mind? see above
            Hx[r_mind : r_mind + 2, r_ind : r_ind + mmsl] = Hp_k

        return Hx
    
    def get_Hw(self, x_pred : np.ndarray, seen_lms, seen_rs) -> np.ndarray:
        """
            Function to get a large jacobian for a measurement.
            May have to be adopted later on
        """
        # -3 because we only have measurements of the objects. So the state gets subtracted
        Hw = np.eye(x_pred.size -3)
        return Hw

    def get_W_est(self, x_len : int) -> np.ndarray:
        """
            Function to return the W matrix of the full measurement
            assumes independent, non-correlated measurements, e.g. block diagonal of self._W_est 
            is ALWAYS the same size as the observation vector -3 for the robot states
        """

        _W = self._W_est
        W = np.kron(np.eye(int(x_len), dtype=int), _W)
        return W

    # functions for extending the map
    def get_g_funcs_lms(self, x_est : np.ndarray, unseen : dict, n : int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        Gx = np.zeros((n, 3))
        Gz = np.zeros((n,  n))
        xf = np.zeros(n)
        xv = x_est[:3]
        for i, (lm_id, z) in enumerate(unseen.items()):
            xf_i = self.sensor.g(xv, z)
            Gz_i = self.sensor.Gz(xv, z)
            Gx_i = self.sensor.Gx(xv, z)

            xf[i*2 : i*2 + 2] = xf_i
            Gz[i*2 : i*2 + 2, i*2 : i*2 + 2] = Gz_i
            Gx[i*2 : i*2 + 2, :] = Gx_i

            # add the landmark
            self._landmark_add(lm_id)
            if self._verbose:
                print(
                f"landmark {lm_id} seen for first time,"
                f" state_idx={self.landmark_index(lm_id)}"
                )
                    
        return xf, Gz, Gx

    def get_g_funcs_rs(self, x_est : np.ndarray, unseen : dict, n : int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # function to check if the motion model is kinematic
        is_kinematic = self.has_kinematic_model()

        Gx = np.zeros((n, 3))
        Gz = np.zeros((n,  n))
        xf = np.zeros(n)
        xv = x_est[:3]
        # state_len = self.get_state_length()  #  figure out which variables here are dependent on state length and which are fixed 2
        mmsl = self.motion_model.state_length
        for i, (r_id, z) in enumerate(unseen.items()):
            xf_i = self.sensor.g(xv, z, is_kinematic)
            Gz_i = self.sensor.Gz(xv, z, is_kinematic)
            Gx_i = self.sensor.Gx(xv, z, is_kinematic)
            
            # use the motion model offsets.
            # in the kinematic case, Gz_i is (4x2), Gx_i is (4x3)
            xf[i*mmsl : i*mmsl + mmsl] = xf_i
            # Gz is a stack where the diagonals are the individual Gz_is. Dimensions are: (n*4) x (nx2)
            Gz[i*mmsl : i*mmsl + mmsl, i*mmsl : i * mmsl + 2] = Gz_i
            # Gx is ((n *mmsl) x 3)
            Gx[i*mmsl : i*mmsl + mmsl, :] = Gx_i

            # add the landmark
            self._robot_add(r_id)
            if self._verbose:
                print(
                f"robot {r_id} seen for first time,"
                f" state_idx={self.robot_index(r_id)}"
                )
                    
        return xf, Gz, Gx

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
            # ! check function from PC - [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/Vehicle.py]] L 643
            od = rob.step(pause=pause)
            rsd[j + self.sensor.robot_offset] = rob.x.copy()
            
        # =================================================================
        # P R E D I C T I O N
        # =================================================================
        Fx = self.get_Fx(odo)
        Fv = self.get_Fv(odo)
        V = self.get_V()
        x_est = self.x_est
        P_est = self.P_est
        rbt = self.robot
        x_pred, P_pred = predict(x_est, P_est, rbt, odo, Fx, Fv, V)
        x_pred = self.predict_robots(x_pred)

        # =================================================================
        # P R O C E S S    O B S E R V A T I O N S
        # =================================================================

        #  read the sensor - get every landmark and every robot
        zk, rk = self.sensor.reading()

        # if baseline models are available - select the zks accordingly
        if self.ekf_exclude is not None:
            zk_exc = zk
        
        if self.ekf_include is not None:
            # robot indices are 100 + r_id
            zk_inc = self.fuse_observations(zk, rk)

        # split the landmarks into seen and unseen
        seen_lms, unseen_lms = self.split_readings(zk, self._isseenbefore)
        seen_rs, unseen_rs = self.split_readings(rk, self._isseenbefore_robot)
        
        if self._verbose:
            [print(f"Robot {r} in map") for r in self.seen_robots]
            [print(f"Currently observed robots: {r}") for r in rk]
        # First - update with the seen
        # ? what if this gets switched around -> is it better to first insert and then update or vice versa? Theoretically the same?
        # get the innovation - includes updating the landmark count
        # ! todo make sure that the innovation only gets called if there are seen lms and / or robots
        # ! else this step is unneccessary
        innov = self.get_innovation(x_pred, seen_lms, seen_rs)
        if innov.size > 0:        
            # get the jacobians
            Hx = self.get_Hx(x_pred, seen_lms, seen_rs)
            Hw = self.get_Hw(x_pred, seen_lms, seen_rs)

            # calculate Covariance innovation, K and the rest
            x_len = int((len(x_pred) - 3) / 2)
            W_est = self.get_W_est(x_len)
            S = calculate_S(P_pred, Hx, Hw, W_est)
            K = calculate_K(Hx, S, P_pred)

            # Updating state and covariance
            x_est = update_state(x_pred, K, innov)
            x_est[2] = base.wrap_mpi_pi(x_est[2])
            if self._joseph:
                P_est = update_covariance_joseph(P_pred, K, W_est, Hx)
            else:
                P_est = update_covariance_normal(P_pred, S, K)
        else:
            P_est = P_pred
            x_est = x_pred
            # for history keeping
            S = None
            K = None
        
        # =================================================================
        # Insert New Landmarks
        # =================================================================

        # Section on extending the map for the next timestep:
        # new landmarks, seen for the first time
        # Inserting new landmarks
        # extend the state vector and covariance
        if unseen_lms:
            W_est = self._W_est     
            n_new = len(unseen_lms) * 2
            W_est_full = self.get_W_est(int(n_new / 2))
            xf, Gz, Gx = self.get_g_funcs_lms(x_est, unseen_lms, n_new)
                
            ### section on adding the lms with the big array
            x_est, P_est = extend_map(
                x_est, P_est, xf, Gz, Gx, W_est_full
            )

        # inserting new robot variables
        if unseen_rs:
            W_est = self._W_est
            n_new = len(unseen_rs) * self.motion_model.state_length
            W_est_full = self.get_W_est(int(n_new / 2))
            xf, Gz, Gx = self.get_g_funcs_rs(x_est, unseen_rs, n_new)
                
            ### section on adding the lms with the big array
            x_est, P_est = extend_map(
                x_est, P_est, xf, Gz, Gx, W_est_full
            )

        # updating the variables before the next timestep
        self._x_est = x_est
        self._P_est = P_est

        # =================================================================
        # Baselines
        # =================================================================

        if self.ekf_exclude is not None:
            t = self.robot._t
            x_exc, P_exc = self.ekf_exclude.step(t, odo, zk_exc)

        if self.ekf_include is not None:
            t = self.robot._t
            x_inc, P_inc = self.ekf_include.step(t, odo, zk_inc)

        if self.ekf_FP is not None:
            t = self.robot._t
            x_fp, P_fp = self.ekf_FP.step(t, odo, zk)

        # logging
        if self._keep_history:
            hist = self._htuple(
                self.robot._t,
                self.robot.x.copy(),
                rsd,
                x_est.copy(),
                odo.copy(),
                P_est.copy(),
                innov.copy() if innov is not None else None,
                S.copy() if S is not None else None,
                K.copy() if K is not None else None,
                zk if zk is not None else None,
                rk if zk is not None else None,
                self.seen_robots,
                self.landmarks
            )
            self._history.append(hist)


    ##### plotting functions from here on out
    def split_states(self, x : np.ndarray, P : np.ndarray, seen_dyn_lms : list | dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            Function to split the state and covariance into a static and dynamic part
            works on pre-removed matrices, e.g. where the robot is already removed
        """
        r_idxs = [self.robot_index(r_id) - 3 for r_id in seen_dyn_lms]
        b_x = np.ones(x.shape, dtype=bool)
        r_state_len = self.motion_model.state_length
        for r_idx in r_idxs:
            b_x[r_idx : r_idx + r_state_len] = False
        x_stat = x[b_x]
        P_stat = P[np.ix_(b_x, b_x)]        
        x_dyn = x[~b_x]
        P_dyn = P[np.ix_(~b_x, ~b_x)]

        return x_stat, P_stat, x_dyn, P_dyn

    def get_robot_xyt(self, r_id : int) -> np.array:
        r"""
        Get estimated vehicle trajectory

        :return: vehicle trajectory where each row is configuration :math:`(x, y, \theta)`
        :rtype: ndarray(n,3)

        :seealso: :meth:`plot_xy` :meth:`run` :meth:`history`
        """
        r_mind = self.robot_index(r_id)
        if self._est_vehicle:
            # todo - correct this - only plot if the robot is already inserted into the map
            xyt = np.array([h.xest[r_mind : r_mind + 2] if (h.xest.size > r_mind) else np.array([0., 0.]) for h in self._history])
        else:
            xyt = None
        return xyt

    def get_start_t_robot(self, robot_id : int):
        """
            Function to get the start time when a robot is in the history
        """
        start_t = None
        r_ind = self.robot_index(robot_id)
        for i, h in enumerate(self.history):
            if len(h.xest) > r_ind:
                start_t = i
                break
        return start_t

    def get_start_t_lm(self, lm_id : int):
        """
            Function to get a start time when a landmark is in the history
        """
        start_t = 0
        lm_ind = self.landmark_index(lm_id)
        for i, h in enumerate(self.history):
            if len(h.xest > lm_ind):
                start_t = i
                break
        return start_t


    ## Evaluation section
    def get_Pnorm(self, k=None):
        """
        Get covariance norm from simulation

        :param k: timestep, defaults to None
        :type k: int, optional
        :return: covariance matrix norm
        :rtype: float or ndarray(n)

        If ``k`` is given return covariance norm from simulation timestep ``k``, else
        return all covariance norms as a 1D NumPy array.

        :seealso: :meth:`get_P` :meth:`run` :meth:`history`
        """
        if k is not None:
            return np.sqrt(np.linalg.det(self._history[k].Pest))
        else:
            p = [np.sqrt(np.linalg.det(h.Pest)) for h in self._history]
            return np.array(p)

    def _filler_func(self, dim : int) -> np.ndarray:
        return np.sqrt(np.linalg.det(-1 * np.ones((dim, dim))))
    
    def _ind_in_P(self, P : np.ndarray, m_ind : int) -> bool:
        return True if P.shape[0] > m_ind else False
    
    def get_Pnorm_map(self, map_ind : int, t : int = None, offset : int  = 2):
        if t is not None:
            P_h = self.history[t].Pest
            P = P_h[map_ind : map_ind + offset, map_ind : map_ind + offset] if self._ind_in_P(P_h, map_ind) else self._filler_func(offset)
            return np.sqrt(np.linalg.det(P))
        else:
            p = [np.sqrt(np.linalg.det(h.Pest[map_ind : map_ind + offset, map_ind : map_ind + offset])) if self._ind_in_P(h.Pest, map_ind) else self._filler_func(offset) for h in self._history]
            return np.array(p)

    def get_Pnorm_lm(self, lm_id : int, t : int  = None):
        ind = self.landmark_index(lm_id)
        return self.get_Pnorm_map(ind, t)
        
    def get_Pnorm_r(self, r_id : int, t : int=None):
        ind = self.robot_index(r_id)
        return self.get_Pnorm_map(ind, t)