from roboticstoolbox import EKF


class EKF_MR(EKF):
    """ inherited class for the multi-robot problem"""


    def __init__(self, robot, sensor=None, map=None, P0=None, x_est=None, joseph=True, animate=True, x0=..., verbose=False, history=True, workspace=None,
                ):
        super().__init__(robot, sensor, map, P0, x_est, joseph, animate, x0, verbose, history, workspace)

        # TODO: append second robot here

        # only worry about SLAM things

    def step(self, pause=None):
        """
            Execute one timestep of the simulation  
        """

        # move the robot
        odo = self.robot.step(pause=pause)

        # =================================================================
        # P R E D I C T I O N
        # =================================================================
        if self._est_vehicle:
            # split the state vector and covariance into chunks for
            # vehicle and map
            xv_est = self._x_est[:3]
            xm_est = self._x_est[3:]
            Pvv_est = self._P_est[:3, :3]
            Pmm_est = self._P_est[3:, 3:]
            Pvm_est = self._P_est[:3, 3:]
        else:
            xm_est = self._x_est
            Pmm_est = self._P_est

        if self._est_vehicle:
            # evaluate the state update function and the Jacobians
            # if vehicle has uncertainty, predict its covariance
            xv_pred = self.robot.f(xv_est, odo)

            Fx = self.robot.Fx(xv_est, odo)
            Fv = self.robot.Fv(xv_est, odo)
            Pvv_pred = Fx @ Pvv_est @ Fx.T + Fv @ self.V_est @ Fv.T
        else:
            # otherwise we just take the true robot state
            xv_pred = self._robot.x

        if self._est_ekf_map:
            if self._est_vehicle:
                # SLAM case, compute the correlations
                Pvm_pred = Fx @ Pvm_est

            Pmm_pred = Pmm_est
            xm_pred = xm_est

        # put the chunks back together again
        if self._est_vehicle and not self._est_ekf_map:
            # vehicle only
            x_pred = xv_pred
            P_pred = Pvv_pred
        elif not self._est_vehicle and self._est_ekf_map:
            # map only
            x_pred = xm_pred
            P_pred = Pmm_pred
        elif self._est_vehicle and self._est_ekf_map:
            # vehicle and map
            x_pred = np.r_[xv_pred, xm_pred]
            # fmt: off
            P_pred = np.block([
                [Pvv_pred,   Pvm_pred], 
                [Pvm_pred.T, Pmm_pred]
            ])
            # fmt: on

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

        if self.sensor is not None:
            #  read the sensor
            z, lm_id = self.sensor.reading()
            sensorReading = z is not None
        else:
            lm_id = None  # keep history saving happy
            z = None
            sensorReading = False

        if sensorReading:
            #  here for MBL, MM, SLAM

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
        else:
            innov = None

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

    


