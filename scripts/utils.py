from typing import Tuple

from roboticstoolbox import EKF, RangeBearingSensor
from roboticstoolbox.mobile import VehicleBase
import numpy as np
from spatialmath import base
from scipy.linalg import block_diag


"""
    ? change the EKF P and x to be a pandas Dataframe multiarray - for simpler indexing
"""

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

    # Full Prediction steps! how they are executed
    def predict_MR_PC(self, odo) -> Tuple[np.ndarray, np.ndarray]:
        """
            function for the predict step of the Multi-Robot case,
            staying true to PC implementation. 
        """
        xv_est = self._x_est[:5]
        xm_est = self.x_est[5:]
        Pvv_est = self._P_est[:5, :5]
        Pmm_est = self._P_est[5:, 5:]
        Pvm_est = self._P_est[:5, 5:]
        # this is how it is normally done
        xv_pred = self.f(xv_est, odo)
        Fx = self.Fx(xv_est, odo)
        Fv = self.Fv(xv_est, odo)
        
        # vechicle predict
        Pvv_pred = Fx @ Pvv_est @ Fx.T + Fv @ self.V_est @ Fv.T

        # SLAM case, compute the correlations
        #! this is not part of the book! - or if it is, it's not very clear
        # but it makes a lot of sense 
        # except for the fact that it is not quadratic?
        # Is this eradicated because it is employed as .T further below? 
        Pvm_pred = Fx @ Pvm_est

        # map state and cov stay the same
        Pmm_pred = Pmm_est
        xm_pred = xm_est

        # vehicle and map
        x_pred = np.r_[xv_pred, xm_pred]
        # fmt: off
        P_pred = np.block([
            [Pvv_pred,   Pvm_pred], 
            [Pvm_pred.T, Pmm_pred]
        ])

        return x_pred, P_pred

    def predict(self, odo, Fx : np.ndarray, Fv : np.ndarray, V : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            prediction function just mathematically
            requires the correct Fx and Fv dimensions
            V is the noise variance matrix. Is constructed by inserting the correct V in the right places.
            which means that for every dynamic landmark, a V must be inserted at the diagonal block. 
        """
        # state prediction - only predict the vehicle in the non-kinematic case. 
        xv_est = self.x_est[:3]
        xm_est = self.x_est[3:]
        xm_pred = xm_est
        xv_pred = self.robot.f(xv_est, odo)
        x_pred = np.r_[xv_pred, xm_pred]
        P_est = self.P_est
        
        # Covariance prediction
        # differs slightly from PC. see
        # [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/EKF.py]]
        # L 806
        P_pred = Fx @ P_est @ Fx.T + Fv @ V @ Fv.T

        return x_pred, P_pred

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

    def calculate_S(self, P_pred : np.ndarray, Hx : np.ndarray, Hw : np.ndarray, W_est : np.ndarray) -> np.ndarray:
        """
            function to calculate S
        """
        S = Hx @ P_pred @ Hx.T + Hw @ W_est @ Hw.T
        return S

    def calculate_K(self, Hx : np.ndarray, S : np.ndarray, P_pred : np.ndarray) -> np.ndarray:
        """
            Function to calculate K
        """
        K = P_pred @ Hx.T @ np.linalg.inv(S)
        return K

    def update_state(self, x_pred : np.ndarray, K : np.ndarray, innov : np.ndarray) -> np.ndarray:
        """
            Function to update the predicted state with the Kalman gain and the innovation
            K or innovation for not measured landmarks should both be 0
        """
        x_est = x_pred + K @ innov
        return x_est
    
    def update_covariance_joseph(self, P_pred : np.ndarray, K : np.ndarray, W_est : np.ndarray, Hx : np.ndarray) -> np.ndarray:
        """
            Function to update the covariance
        """
        I = np.eye(P_pred.shape[0])
        P_est = (I - K @ Hx) @ P_pred @ (I - K @ Hx).T + K @ W_est @ K.T
        return P_est
    
    def update_covariance_normal(self, P_pred : np.ndarray, S : np.ndarray, K : np.ndarray) -> np.ndarray:
        """
            update covariance
        """
        P_est = P_pred - K @ S @ K.T
        # symmetry enforcement
        P_est = 0.5 * (P_est + P_est.T)
        return P_est
    
    def get_Hx(self, x_pred : np.ndarray, seen_lms, seen_rs) -> np.ndarray:
        """
            Function to get a large state measurement jacobian
            the jacobian here is not 2x... but (2xzk) x ... . See Fenwick (2.24)
        """
        # set up a large zero-matrix
        dim = x_pred.size 
        Hx = np.zeros((dim -1, dim))    # one less row, because vehicle Hx is 2x3
        xv_pred = x_pred[:3]
        # go through all static landmarks
        for lm_id, _ in seen_lms:
            # get the landmark index in the map and the state estimate
            l_ind = self.landmark_mindex(lm_id)
            xf = x_pred[l_ind : l_ind + 2]
            # calculate BOTH jacobians with respect to the current position and estimate and state estimate
            Hp_k = self.sensor.Hp(xv_pred, xf)
            Hxv = self.sensor.Hx(xv_pred, xf)
            # insert the vehicle Jacobian in the first COLUMN - corresponding to the first three states
            Hx[l_ind : l_ind+2, :3] = Hxv
            Hx[l_ind : l_ind+2, l_ind : l_ind+2] = Hp_k 

        # go through all dynamic landmarks
        for r_id, _ in seen_rs:
            # get robot index
            r_ind = self.robot_index(r_id)
            xf = x_pred[r_ind : r_ind + 2]
            # calculate BOTH jacobians with respect to the current position and estimate and state estimate
            Hp_k = self.sensor.Hp(xv_pred, xf)
            Hxv = self.sensor.Hx(xv_pred, xf)
            # insert the vehicle Jacobian in the first COLUMN - corresponding to the first three states
            Hx[r_ind : r_ind+2, :3] = Hxv
            Hx[r_ind : r_ind+2, r_ind : r_ind+2] = Hp_k

        return Hx
    
    def get_Hw(self, x_pred : np.ndarray, seen_lms, seen_rs) -> np.ndarray:
        """
            Function to get a large jacobian for a measurement.
            May have to be adopted later on
        """
        Hw = np.eye(x_pred.shape[0])
        return Hw

    ######## Extending the Map section
    def _extend_map(self, P : np.ndarray, x : np.ndarray, z : np.ndarray, W_est) -> Tuple[np.ndarray, np.ndarray]:
        # this is a new landmark, we haven't seen it before
        # estimate position of landmark in the world based on
        # noisy sensor reading and current vehicle pose

        # M = None
        xv = x[:3]
        xm = x[3:]

        # estimate its position in the world frame based on observation and vehicle state
        xf = self.sensor.g(xv, z)

        # append this estimate to the state vector
        x_ext = np.r_[xv, xm, xf]

        # get the Jacobian for the new landmark
        Gz = self.sensor.Gz(xv, z)

        # extend the covariance matrix
        n = len(x)
        # estimating vehicle state
        Gx = self.sensor.Gx(xv, z)
        Yz = np.block([
            [np.eye(n), np.zeros((n, 2))    ],
            [Gx,        np.zeros((2, n-3)), Gz]
        ])
        P_ext = Yz @ block_diag(P, W_est) @ Yz.T

        return x_ext, P_ext

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
        x_pred, P_pred = self.predict(odo, Fx, Fv, V)

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
            Hx = self.get_Hx(seen_lms, seen_rs)
            Hw = self.get_Hw(seen_lms, seen_rs)

            # calculate Covariance innovation, K and the rest
            S = self.calculate_S(P_pred, Hx, Hw, self._W_est)
            K = self.calculate_K(Hx, S, P_pred)

            # Updating state and covariance
            x_est = self.update_state(x_pred, K, innov)
            x_est[2] = base.wrap_mpi_pi(x_est[2])
            if self._joseph:
                P_est = self.update_covariance_joseph(P_pred, K, self._W_est, Hx)
            else:
                P_est = self.update_covariance_normal(P_pred, S, K)
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
        W_est = self._W_est
        for lm_id, z in unseen_lms.items(): 
            x_est, P_est = self._extend_map(
                P_est, x_est, z, W_est
            )
            self._landmark_add(lm_id)
            if self._verbose:
                print(
                f"landmark {lm_id} seen for first time,"
                f" state_idx={self.landmark_index(lm_id)}"
                )

        # inserting new robot variables
        for r_id, z in unseen_rs.items():
            x_est, P_est = self._extend_map(
                P_est, x_est, z, W_est
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
