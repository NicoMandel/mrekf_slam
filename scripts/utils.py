from typing import Tuple

from roboticstoolbox import EKF, RangeBearingSensor
from roboticstoolbox.mobile import VehicleBase
import numpy as np
from spatialmath import base
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


class RobotSensor(RangeBearingSensor):
    """
        Inherit a Range and Bearing Sensor
        senses map and other robots
    """

    def __init__(self, robot, r2 : list, map, line_style=None, poly_style=None, covar=None, range=None, angle=None, plot=False, seed=0, **kwargs):
        if not isinstance(r2, list):
            raise TypeError("Robots should be a list of other robots. Of the specified robot format")
        self._r2s = r2
        super().__init__(robot, map=map, line_style=line_style, poly_style=poly_style, covar=covar, range=range, angle=angle, plot=plot, seed=seed, **kwargs)
        
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

class EKF_MR(EKF):
    """ inherited class for the multi-robot problem"""

    def __init__(self, robot, r2 : list, V2, sensor : RobotSensor =None, map=None, P0=None, x_est=None, joseph=True, animate=True, x0=[0, 0, 0], verbose=False, history=True, workspace=None,
                ):
        super().__init__(robot, sensor=sensor, map=map, P0=P0, x_est=x_est, joseph=joseph, animate=animate, x0=x0, verbose=verbose, history=history, workspace=workspace)
        # Calling arguments:
        # robot=(robot, V), P0=P0, sensor=(sensor, W) ! sensor is now a sensor that also detects the other robot
        if not isinstance(r2, list):
            raise TypeError("r2 must be a list. Must also be tuple (vehicle, V_est)")
        
        # list of robots. and needs list of seen robots
        self._robots = r2
        self._seen_robots = {}

        # Management of the model that the agent has of the other
        if V2.shape != (2, 2):
            raise ValueError("vehicle state covariance V_est must be 2x2")
        self._V_model = V2

        # self._W_est is the estimate of the sensor covariance

        # only worry about SLAM things 
        # Sensor init should be the same [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/EKF.py]]

        # map init should be the same - L: 264
        
        # P0 init is the same

        # case handling for which cases is irrelevant
        
        # robot initialization of P_est and robot x is the same

        # self.init() - no underscores - should also be the same - l .605
        # robot init appears to be the same - for vehicle base case
        # sensor init also - for sensorBase case


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
    def V_model(self):
        return self._V_model

    ######## Section on Landmark and Robot Management
    def get_state_length(self):
        """
            Function to get the current length of the state vector - to figure out where to append
        """
        return 3 + 2 * len(self._seen_robots) + 2 * len(self.landmarks)

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

    # simple model prediction and jacobian for constant location case
    def f(self, x):
        """
            function to replicate f for the multi-robot case.
            In PC: 
            [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/Vehicle.py]]
            line 188 ff. does not add any noise
            f(x_k+1) = x_k + v_x
            same for y
        """
        return x

    def Fx(self):
        """
            In case of a constant location model, the Fx is just an Identity matrix
            f(x_k+1) = x_k + v_x
        """
        fx = np.eye(2,2)
        return fx

    def Fv(self):
        """
            In case of a constant location model, the Fv is just an Identity matrix
            f(x_k+1) = x_k + v_x
        """
        fv = np.eye(2,2)
        return fv
    
    # model prediction and jacobians for constant velocity case.
    # see https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/
    def f_kinematic(self, x, dt):
        pass

    def Fx_kinematic(self, x):
        pass
    
    def Fv_kinematic(self,x):
        pass    

    ###### helper functions for getting the right matrices
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
        for r in self.seen_robots or []:      # careful - robots is not seen robots!
            r_ind = self.robot_index(r)
            Fx[r_ind : r_ind + 2, r_ind : r_ind +2] = self.Fx()

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
        for r in self.seen_robots or []:      # careful - robots is not seen robots!
            r_ind = self.robot_index(r)
            ins_r = r_ind
            ins_c = r_ind - 1
            Fv[ins_r : ins_r + 2, ins_c : ins_c + 2] = self.Fv()

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
        for r in self.seen_robots or []:
            # is one less, because the state for the robot is 3x3, but the noise model is only 2x2
            r_ind = self.robot_index(r) -1
            Vm[r_ind : r_ind + 2, r_ind : r_ind + 2] = self.V_model

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

    def get_innovation(self, x_pred : np.ndarray, seen_lms, seen_rs) -> np.ndarray:
        """
            Function to calculate the innovation
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

    def get_Hx(self, x_pred : np.ndarray, seen_lms, seen_rs) -> np.ndarray:
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
            Hx[l_mind : l_mind+2, l_mind : l_mind+2] = Hp_k 

        # go through all dynamic landmarks
        for r_id, _ in seen_rs.items():
            # get robot index
            r_ind = self.robot_index(r_id)
            xf = x_pred[r_ind : r_ind + 2]
            # calculate BOTH jacobians with respect to the current position and estimate and state estimate
            Hp_k = self.sensor.Hp(xv_pred, xf)
            Hxv = self.sensor.Hx(xv_pred, xf)
            # insert the vehicle Jacobian in the first COLUMN - corresponding to the first three states
            r_mind = r_ind - 3       # see above
            Hx[r_mind : r_mind+2, :3] = Hxv
            # robot index is 1 row before, because Hxv is only 2 rows
            Hx[r_mind : r_mind+2, r_mind : r_mind+2] = Hp_k

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
            is ALWAYS the same size as the state vector -3 for the robot states
        """

        _W = self._W_est
        W = np.kron(np.eye(int(x_len), dtype=int), _W)
        return W

    # functions for extending the map
    def get_g_funcs(self, x_est : np.ndarray, unseen : dict, n : int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
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

    def get_Gx(self, unseen : dict) -> np.ndarray:
        pass

    def step(self, pause=None):
        """
            Execute one timestep of the simulation
            # original in file [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/EKF.py]]
            Line 773 
        """
        # move the robot
        odo = self.robot.step(pause=pause)
        for robot in self.robots:
            od = robot.step(pause=pause)        # todo: include this into the logging?
        # =================================================================
        # P R E D I C T I O N
        # =================================================================
        Fx = self.get_Fx(odo)
        Fv = self.get_Fv(odo)
        V = self.get_V()
        x_est = self.x_est
        P_est = self.P_est
        rbt = self.robot
        x_pred, P_pred = EKF_base.predict(x_est, P_est, rbt, odo, Fx, Fv, V)

        # =================================================================
        # P R O C E S S    O B S E R V A T I O N S
        # =================================================================

        #  read the sensor - get every landmark and every robot
        zk, rk = self.sensor.reading()

        # split the landmarks into seen and unseen
        seen_lms, unseen_lms = self.split_readings(zk, self._isseenbefore)
        seen_rs, unseen_rs = self.split_readings(rk, self._isseenbefore_robot)

        # First - update with the seen
        # ? what if this gets switched around -> is it better to first insert and then update or vice versa? Theoretically the same?
        # get the innovation - includes updating the landmark count
        innov = self.get_innovation(x_pred, seen_lms, seen_rs)
        if innov.size > 0:        
            # get the jacobians
            Hx = self.get_Hx(x_pred, seen_lms, seen_rs)
            Hw = self.get_Hw(x_pred, seen_lms, seen_rs)

            # calculate Covariance innovation, K and the rest
            x_len = int((len(x_pred) - 3) / 2)
            W_est = self.get_W_est(x_len)
            S = EKF_base.calculate_S(P_pred, Hx, Hw, W_est)
            K = EKF_base.calculate_K(Hx, S, P_pred)

            # Updating state and covariance
            x_est = EKF_base.update_state(x_pred, K, innov)
            x_est[2] = base.wrap_mpi_pi(x_est[2])
            if self._joseph:
                P_est = EKF_base.update_covariance_joseph(P_pred, K, W_est, Hx)
            else:
                P_est = EKF_base.update_covariance_normal(P_pred, S, K)
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
        # ! TODO: ONLY IF THERE ARE UNSEEN ITEMS - check
        W_est = self._W_est     # this time only using it once
        n_new = len(unseen_lms) * 2
        W_est_full = np.kron(np.eye(int(n_new / 2), dtype=int), W_est)
        xf_full, Gz_full, Gx_full = self.get_g_funcs(x_est, unseen_lms, n_new)

        x_est_full = x_est
        P_est_full = P_est
        for i, (lm_id, z) in enumerate(unseen_lms.items()):
            xv = x_est[:3]
            xf = self.sensor.g(xv, z)
            Gz = self.sensor.Gz(xv, z)
            Gx = self.sensor.Gx(xv, z)

            x_est, P_est = EKF_base.extend_map(
                x_est, P_est, xf, Gz, Gx, W_est,
            )
            # self._landmark_add(lm_id)
            
        ### section on adding the lms with the big array
        x_est_full, P_est_full = EKF_base.extend_map(
            x_est_full, P_est_full, xf_full, Gz_full, Gx_full, W_est_full
        )

        # print("Test Debug line")
        assert np.allclose(x_est, x_est_full)
        assert np.allclose(P_est, P_est_full)
        # inserting new robot variables
        # todo: do the same combined update step for the unseen robots
        for r_id, z in unseen_rs.items():
            xv = x_est[:3]
            xf = self.sensor.g(xv, z)
            Gz = self.sensor.Gz(xv, z)
            Gx = self.sensor.Gx(xv, z)

            x_est, P_est = EKF_base.extend_map(
                x_est, P_est, xf, Gz, Gx, W_est
            )
            self._robot_add(r_id)
            if self._verbose:
                print(
                f"robot {r_id} seen for first time,"
                f" state_idx={self.robot_index(r_id)}"
                )

        # updating the variables before the next timestep
        self._x_est = x_est
        self._P_est = P_est

        # logging issues
        lm_id = None
        z = zk
        if self._keep_history:
            hist = self._htuple(
                self.robot._t,
                x_est.copy(),
                odo.copy(),
                P_est.copy(),
                innov.copy() if innov is not None else None,
                S.copy() if S is not None else None,
                K.copy() if K is not None else None,
                lm_id if lm_id is not None else -1,
                z.copy() if z is not None else None,
            )
            self._history.append(hist)

    # Plotting stuff
    def plot_map(self, marker=None, ellipse=None, confidence=0.95, block=None):
        """
        Plot estimated landmarks

        :param marker: plot marker for landmark, arguments passed to :meth:`~matplotlib.axes.Axes.plot`, defaults to "r+"
        :type marker: dict, optional
        :param ellipse: arguments passed to :meth:`~spatialmath.base.graphics.plot_ellipse`, defaults to None
        :type ellipse: dict, optional
        :param confidence: ellipse confidence interval, defaults to 0.95
        :type confidence: float, optional
        :param block: hold plot until figure is closed, defaults to None
        :type block: bool, optional

        Plot a marker  and covariance ellipses for each estimated landmark.

        :seealso: :meth:`get_map` :meth:`run` :meth:`history`
        """
        if marker is None:
            marker_stat = {
                "marker": "+",
                "markersize": 10,
                "markerfacecolor": "b",
                "linewidth": 0,
            }
            marker_dyn = {
                "marker": "x",
                "markersize": 10,
                "markerfacecolor": "r",
                "linewidth": 0,
            }

        # todo split into static and dynamic landmarks here for plotting
        x_stat = []

        xm = self._x_est
        P = self._P_est
        
        xm = xm[3:]
        P = P[3:, 3:]

        # mark the estimate as a point
        xm = xm.reshape((-1, 2))  # arrange as Nx2
        plt.plot(xm[:, 0], xm[:, 1], label="estimated landmark", **marker_stat)

        # add an ellipse
        if ellipse is not None:
            for i in range(xm.shape[0]):
                Pi = self.P_est[i : i + 2, i : i + 2]
                # put ellipse in the legend only once
                if i == 0:
                    base.plot_ellipse(
                        Pi,
                        centre=xm[i, :],
                        confidence=confidence,
                        inverted=True,
                        label=f"{confidence*100:.3g}% confidence",
                        **ellipse,
                    )
                else:
                    base.plot_ellipse(
                        Pi,
                        centre=xm[i, :],
                        confidence=confidence,
                        inverted=True,
                        **ellipse,
                    )
        # plot_ellipse( P * chi2inv_rtb(opt.confidence, 2), xf, args{:});
        if block is not None:
            plt.show(block=block)

### standard EKF algorithm that just does the prediction and the steps
class EKF_base(object):
    """
        basic EKF algorithm that just does the mathematical steps
        mathematical methods for the normal steps are exposed as class methods so that they can be reused by the MR_EKF
    """

    def __init__(self, x0 : np.ndarray = None, P0 : np.ndarray = None, V : np.ndarray = None, W : np.ndarray = None) -> None:
        self.x0 = x0
        self.P0 = P0
        # base models
        assert V.shape == (2,2), "Vehicle Covariance must be 2x2"

        self.W = W
        self.V = V
    
    ### section with static methods - pure mathematics, just gets used by every instance
    @staticmethod
    def predict(x_est : np.ndarray, P_est : np.ndarray, robot : VehicleBase, odo, Fx : np.ndarray, Fv : np.ndarray, V : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            prediction function just mathematically
            requires the correct Fx and Fv dimensions
            V is the noise variance matrix. Is constructed by inserting the correct V in the right places.
            which means that for every dynamic landmark, a V must be inserted at the diagonal block. 
        """
        # state prediction - only predict the vehicle in the non-kinematic case. 
        xv_est = x_est[:3]
        xm_est = x_est[3:]
        xm_pred = xm_est
        xv_pred = robot.f(xv_est, odo)
        x_pred = np.r_[xv_pred, xm_pred]
        P_est = P_est
        
        # Covariance prediction
        # differs slightly from PC. see
        # [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/EKF.py]]
        # L 806
        P_pred = Fx @ P_est @ Fx.T + Fv @ V @ Fv.T

        return x_pred, P_pred

    ### update steps
    @staticmethod
    def calculate_S(P_pred : np.ndarray, Hx : np.ndarray, Hw : np.ndarray, W_est : np.ndarray) -> np.ndarray:
        """
            function to calculate S
        """
        S = Hx @ P_pred @ Hx.T + Hw @ W_est @ Hw.T
        return S
    
    @staticmethod
    def calculate_K(Hx : np.ndarray, S : np.ndarray, P_pred : np.ndarray) -> np.ndarray:
        """
            Function to calculate K
        """
        K = P_pred @ Hx.T @ np.linalg.inv(S)
        return K
    
    @staticmethod
    def update_state(x_pred : np.ndarray, K : np.ndarray, innov : np.ndarray) -> np.ndarray:
        """
            Function to update the predicted state with the Kalman gain and the innovation
            K or innovation for not measured landmarks should both be 0
        """
        x_est = x_pred + K @ innov
        return x_est

    @staticmethod
    def update_covariance_joseph(P_pred : np.ndarray, K : np.ndarray, W_est : np.ndarray, Hx : np.ndarray) -> np.ndarray:
        """
            Function to update the covariance
        """
        I = np.eye(P_pred.shape[0])
        P_est = (I - K @ Hx) @ P_pred @ (I - K @ Hx).T + K @ W_est @ K.T
        return P_est
    
    @staticmethod
    def update_covariance_normal(P_pred : np.ndarray, S : np.ndarray, K : np.ndarray) -> np.ndarray:
        """
            update covariance
        """
        P_est = P_pred - K @ S @ K.T
        # symmetry enforcement
        P_est = 0.5 * (P_est + P_est.T)
        return P_est
    
    ### inserting new variables
    #todo test this function with just one lm input or with many inputs
    @staticmethod
    def extend_map(x : np.ndarray, P : np.ndarray, xf : np.ndarray, Gz : np.ndarray, Gx : np.ndarray, W_est : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # extend the state vector with the new features
        x_ext = np.r_[x, xf]

        # extend the map
        n_x = len(x)
        n_lm = len(xf)
        Yz = np.block([
            [np.eye(n_x), np.zeros((n_x, n_lm))    ],
            [Gx,        np.zeros((n_lm, n_x-3)), Gz]
        ])
        P_ext = Yz @ block_diag(P, W_est) @ Yz.T

        return x_ext, P_ext