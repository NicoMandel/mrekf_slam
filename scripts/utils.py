from typing import Tuple

from roboticstoolbox import EKF, RangeBearingSensor
from roboticstoolbox.mobile import VehicleBase
import numpy as np
from spatialmath import base

class RobotSensor(RangeBearingSensor):
    """
        Inherit a Range and Bearing Sensor
        does nto sense map, but just other robots

    """

    def __init__(self, robot, r2, map, line_style=None, poly_style=None, covar=None, range=None, angle=None, plot=False, seed=0, **kwargs):
        map = r2 #! ensure this is set correct in the RangeBearing Class
        super().__init__(robot, map, line_style, poly_style, covar, range, angle, plot, seed, **kwargs)

class EKF_MR(EKF):
    """ inherited class for the multi-robot problem"""

    def __init__(self, robot, r2, sensor=None, robotsensor=None, map=None, P0=None, x_est=None, joseph=True, animate=True, x0=..., verbose=False, history=True, workspace=None,
                ):
        super().__init__(robot, sensor, map, P0, x_est, joseph, animate, x0, verbose, history, workspace)
        # Calling arguments:
        # robot=(robot, V), P0=P0, sensor=(sensor, W)
        if r2 is not None:
            if (
                not isinstance(robot, tuple)
                or len(robot) != 2
                or not isinstance(robot[0], VehicleBase)
            ):
                raise TypeError("robot must be tuple (vehicle, V_est)")
        
        self._r2 = r2[0]  # reference to the robot vehicle
        self._V2_est = r2[1]  # estimate of vehicle state covariance V

        self._r2_seen = False
        assert robotsensor is not None, "No sensor detecting other robot. Please verify"
        self._robotsensor = robotsensor
        # only worry about SLAM things 
        # Sensor init should be the same [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/EKF.py]]

        # map init should be the same - L: 264
        
        # P0 init is the same

        # case handling for which cases is irrelevant
        
        # robot initialization of P_est and robot x is the same

        # self.init() - no underscores - should also be the same - l .605
        # robot init appears to be the same - for vehicle base case
        # sensor init also - for sensorBase case

    @property
    def r2(self):
        return self._r2
    
    @property
    def V2_est(self):
        return self._V2_est

    @property
    def r2_seen(self):
        return self._r2_seen

    @property
    def robotsensor(self):
        return self._robotsensor
    
    def f(self, xv_est : np.ndarray, odo):
        """
            function to replicate f for the multi-robot case
        """
        xv_r = self.robot.f(xv_est[:3], odo)
        xv_pred = xv_est.copy(deep=True)
        xv_pred[:3] = xv_r
        return xv_pred

    def Fx(self, xv_est : np.ndarray, odo):
        Fx_r1 = self.robot.Fx(xv_est, odo)
        fx = np.block([
            [Fx_r1,         np.zeros((2,2))],
            [np.zeros((2,3)), np.eye(2)]
        ])
        return fx

    def Fv(self, xv_est : np.ndarray, odo):
        Fv_r1 = self.robot.Fv(xv_est, odo)
        fv = np.block([
            [Fv_r1, np.zeros((3,2))],
            [np.zeros((2,2)), np.eye(2)]
        ])
        return fv
    
    def predict_MR(self, odo) -> Tuple[np.ndarray, np.ndarray]:
        """
            function for the predict step of the Multi-Robot case
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


    def step(self, pause=None):
        """
            Execute one timestep of the simulation
            # original in file [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/EKF.py]]
            Line 773 
        """

        # move the robot
        odo1 = self.robot.step(pause=pause)
        odo2 = self.r2.step(pause=pause)
        # =================================================================
        # P R E D I C T I O N
        # =================================================================

        # split the state vector and covariance into chunks for
        # vehicle and map
        if self.r2_seen:
            xv_est = self._x_est[:5]
            xm_est = self.x_est[5:]
            Pvv_est = self._P_est[:5, :5]
            Pmm_est = self._P_est[5:, 5:]
            Pvm_est = self._P_est[:5, 5:]
        else:        
            xv_est = self._x_est[:3]
            xm_est = self._x_est[3:]
            Pvv_est = self._P_est[:3, :3]
            Pmm_est = self._P_est[3:, 3:]
            Pvm_est = self._P_est[:3, 3:]

        if self.r2_seen:
            x_pred, P_pred = self.predict_MR(x_est, xm_est, Pvv_est, Pmm_est, Pvm_est)
        else:
            # evaluate the state update function and the Jacobians
            xv_pred = self.robot.f(xv_est[:3], odo1)
            
            Fx = self.robot.Fx(xv_est, odo1)
            Fv = self.robot.Fv(xv_est, odo1)
            Pvv_pred = Fx @ Pvv_est @ Fx.T + Fv @ self.V_est @ Fv.T

            Pvm_pred = Fx @ Pvm_est

            Pmm_pred = Pmm_est
            xm_pred = xm_est

            # vehicle and map
            x_pred = np.r_[xv_pred, xm_pred]
            # fmt: off
            P_pred = np.block([
                [Pvv_pred,   Pvm_pred], 
                [Pvm_pred.T, Pmm_pred]
            ])

        # TODO: CONTINUE HERE


        # at this point we have:
        #   xv_pred the state of the vehicle to use to
        #           predict observations
        #   xm_pred the state of the map
        #   x_pred  the full predicted state vector
        #   P_pred  the full predicted covariance matrix

        # initialize the variables that might be computed during
        # the update phase

        doUpdatePhase = False

        # disp('x_pred:') x_pred'

        # =================================================================
        # P R O C E S S    O B S E R V A T I O N S
        # =================================================================
        # TODO: insert the stuff for sensing the other robot 
    

        #  read the sensor
        z, lm_id = self.sensor.reading()
        sensorReading = z is not None

        # compute the innovation
        z_pred = self.sensor.h(xv_pred, lm_id)
        innov = np.array([z[0] - z_pred[0], base.wrap_mpi_pi(z[1] - z_pred[1])])

        if self._est_ekf_map:
            # the ekf_map is estimated MM or SLAM case
            if self._isseenbefore(lm_id):
                # landmark is previously seen

                # get previous estimate of its state
                jx = self.landmark_mindex(lm_id)
                xf = xm_pred[jx : jx + 2]

                # compute Jacobian for this particular landmark
                # xf = self.sensor.g(xv_pred, z) # HACK
                Hx_k = self.sensor.Hp(xv_pred, xf)

                z_pred = self.sensor.h(xv_pred, xf)
                innov = np.array(
                    [z[0] - z_pred[0], base.wrap_mpi_pi(z[1] - z_pred[1])]
                )

                #  create the Jacobian for all landmarks
                Hx = np.zeros((2, len(xm_pred)))
                Hx[:, jx : jx + 2] = Hx_k

                Hw = self.sensor.Hw(xv_pred, xf)

                if self._est_vehicle:
                    # concatenate Hx for for vehicle and ekf_map
                    Hxv = self.sensor.Hx(xv_pred, xf)
                    Hx = np.block([Hxv, Hx])

                self._landmark_increment(lm_id)  # update the count
                if self._verbose:
                    print(
                        f"landmark {lm_id} seen"
                        f" {self._landmark_count(lm_id)} times,"
                        f" state_idx={self.landmark_index(lm_id)}"
                    )
                doUpdatePhase = True

            else:
                # new landmark, seen for the first time

                # extend the state vector and covariance
                x_pred, P_pred = self._extend_map(
                    P_pred, xv_pred, xm_pred, z, lm_id
                )
                # if lm_id == 17:
                #     print(P_pred)
                #     # print(x_pred[-2:], self._sensor._map.landmark(17), base.norm(x_pred[-2:] - self._sensor._map.landmark(17)))

                self._landmark_add(lm_id)
                if self._verbose:
                    print(
                        f"landmark {lm_id} seen for first time,"
                        f" state_idx={self.landmark_index(lm_id)}"
                    )
                doUpdatePhase = False

        else:
            # LBL
            Hx = self.sensor.Hx(xv_pred, lm_id)
            Hw = self.sensor.Hw(xv_pred, lm_id)
            doUpdatePhase = True


        # doUpdatePhase flag indicates whether or not to do
        # the update phase of the filter
        #
        #  DR                        always false
        #  map-based localization    if sensor reading
        #  map creation              if sensor reading & not first
        #                              sighting
        #  SLAM                      if sighting of a previously
        #                              seen landmark

        if doUpdatePhase:
            # disp('do update\n')
            # #  we have innovation, update state and covariance
            #  compute x_est and P_est

            # compute innovation covariance
            S = Hx @ P_pred @ Hx.T + Hw @ self._W_est @ Hw.T

            # compute the Kalman gain
            K = P_pred @ Hx.T @ np.linalg.inv(S)

            # update the state vector
            x_est = x_pred + K @ innov

            if self._est_vehicle:
                #  wrap heading state for a vehicle
                x_est[2] = base.wrap_mpi_pi(x_est[2])

            # update the covariance
            if self._joseph:
                #  we use the Joseph form
                I = np.eye(P_pred.shape[0])
                P_est = (I - K @ Hx) @ P_pred @ (I - K @ Hx).T + K @ self._W_est @ K.T
            else:
                P_est = P_pred - K @ S @ K.T
                # enforce P to be symmetric
                P_est = 0.5 * (P_est + P_est.T)
        else:
            # no update phase, estimate is same as prediction
            x_est = x_pred
            P_est = P_pred
            S = None
            K = None

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

    


