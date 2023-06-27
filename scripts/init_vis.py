"""
    Initial visualisation script
"""
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

# PC imports
import RVC3 as rvc
from roboticstoolbox import Bicycle, LandmarkMap, EKF, VehicleBase, RandomPath

class TreeNode:

    def __init__(self) -> None:
        pass
    
    def add_child(self, other):
        """
            Adding a child of a tree at distance d
        """
    
    def predict(self):
        """
            Apply the update to the child
        """
        for child in self.children:
            child.predict()

    

class Tree:
    def __init__(self, root) -> None:
        self.root = root
    

class Bike:
    def __init__(self, L : float = 1, steer_max : float = 0.45 * np.pi) -> None:
        self.L = L
        self.steer_max = steer_max
    
    def init_state(self, x = 0, y = 0, phi = 0):
        """
            Setting the initial states for the bicycle.
            x, y and 0
        """
        self.x = np.asarray([x, y, phi])

    def step(self, v, phi):
        """
            Applying one TRUE time step to the state. No noise. Is equation 6.1 of the book
            also in f() of chapter 6
        """
        x_o = self.x[0]
        y_o = self.x[1]
        phi_o = self.x[2]
        x_n = x_o + v * np.cos(phi_o)
        y_n = y_o + v * np.sin(phi_o)
        phi_n = phi_o + phi
        self.x = np.asarray([x_n, y_n, phi_n])
        return self.x

def ellipse(
        E: R2x2,
        centre: Optional[ArrayLike2] = (0, 0),
        scale: Optional[float] = 1,
        confidence: Optional[float] = None,
        resolution: Optional[int] = 40,
        inverted: Optional[bool] = False,
        closed: Optional[bool] = False,
    ) -> Points2:
        r"""
        Points on ellipse

        :param E: ellipse
        :type E: ndarray(2,2)
        :param centre: ellipse centre, defaults to (0,0,0)
        :type centre: tuple, optional
        :param scale: scale factor for the ellipse radii
        :type scale: float
        :param confidence: if E is an inverse covariance matrix plot an ellipse
            for this confidence interval in the range [0,1], defaults to None
        :type confidence: float, optional
        :param resolution: number of points on circumferance, defaults to 40
        :type resolution: int, optional
        :param inverted: if :math:`\mat{E}^{-1}` is provided, defaults to False
        :type inverted: bool, optional
        :param closed: perimeter is closed, last point == first point, defaults to False
        :type closed: bool
        :raises ValueError: [description]
        :return: points on circumference
        :rtype: ndarray(2,N)

        The ellipse is defined by :math:`x^T \mat{E} x = s^2` where :math:`x \in
        \mathbb{R}^2` and :math:`s` is the scale factor.

        .. note:: For some common cases we require :math:`\mat{E}^{-1}`, for example
            - for robot manipulability
            :math:`\nu (\mat{J} \mat{J}^T)^{-1} \nu` i
            - a covariance matrix
            :math:`(x - \mu)^T \mat{P}^{-1} (x - \mu)`
            so to avoid inverting ``E`` twice to compute the ellipse, we flag that
            the inverse is provided using ``inverted``.
        """
        from scipy.linalg import sqrtm

        if E.shape != (2, 2):
            raise ValueError("ellipse is defined by a 2x2 matrix")

        if confidence:
            from scipy.stats.distributions import chi2

            # process the probability
            s = math.sqrt(chi2.ppf(confidence, df=2)) * scale
        else:
            s = scale

        xy = circle(resolution=resolution, closed=closed)  # unit circle

        if not inverted:
            E = np.linalg.inv(E)

        e = s * sqrtm(E) @ xy + np.array(centre, ndmin=2).T
        return e

def plot_ellipse(
        E: R2x2,
        centre: ArrayLike2,
        *fmt: Optional[str],
        scale: Optional[float] = 1,
        confidence: Optional[float] = None,
        resolution: Optional[int] = 40,
        inverted: Optional[bool] = False,
        ax: Optional[plt.Axes] = None,
        filled: Optional[bool] = False,
        **kwargs,
    ) -> List[plt.Artist]:
        r"""
        Plot an ellipse using matplotlib

        :param E: matrix describing ellipse
        :type E: ndarray(2,2)
        :param centre: centre of ellipse, defaults to (0, 0)
        :type centre: array_like(2), optional
        :param scale: scale factor for the ellipse radii
        :type scale: float
        :param resolution: number of points on circumferece, defaults to 40
        :type resolution: int, optional
        :return: the matplotlib object
        :rtype: Line2D or Patch.Polygon

        The ellipse is defined by :math:`x^T \mat{E} x = s^2` where :math:`x \in
        \mathbb{R}^2` and :math:`s` is the scale factor.

        .. note:: For some common cases we require :math:`\mat{E}^{-1}`, for example
            - for robot manipulability
            :math:`\nu (\mat{J} \mat{J}^T)^{-1} \nu` i
            - a covariance matrix
            :math:`(x - \mu)^T \mat{P}^{-1} (x - \mu)`
            so to avoid inverting ``E`` twice to compute the ellipse, we flag that
            the inverse is provided using ``inverted``.

        Returns a set of ``resolution``  that lie on the circumference of a circle
        of given ``center`` and ``radius``.

        Example:

            >>> from spatialmath.base import plotvol2, plot_ellipse
            >>> plotvol2(5)
            >>> plot_ellipse(np.array([[1, 1], [1, 2]]), [0,0], 'r')  # red ellipse
            >>> plot_ellipse(np.array([[1, 1], [1, 2]]), [1, 2], 'b--')  # blue dashed ellipse
            >>> plot_ellipse(np.array([[1, 1], [1, 2]]), [-2, -1], filled=True, facecolor='y')  # yellow filled ellipse

        .. plot::

            from spatialmath.base import plotvol2, plot_ellipse
            ax = plotvol2(5)
            plot_ellipse(np.array([[1, 1], [1, 2]]), [0,0], 'r')  # red ellipse
            plot_ellipse(np.array([[1, 1], [1, 2]]), [1, 2], 'b--')  # blue dashed ellipse
            plot_ellipse(np.array([[1, 1], [1, 2]]), [-2, -1], filled=True, facecolor='y')  # yellow filled ellipse
            ax.grid()
        """
        # allow for centre[2] to plot ellipse in a plane in a 3D plot

        xy = ellipse(E, centre, scale, confidence, resolution, inverted, closed=True)
        ax = axes_logic(ax, 2)
        if filled:
            patch = plt.Polygon(xy.T, **kwargs)
            ax.add_patch(patch)
        else:
            plt.plot(xy[0, :], xy[1, :], *fmt, **kwargs)

def plot_ellipse(self, confidence=0.95, N=10, block=None, **kwargs):
    """
    Plot uncertainty ellipses

    :param confidence: ellipse confidence interval, defaults to 0.95
    :type confidence: float, optional
    :param N: number of ellipses to plot, defaults to 10
    :type N: int, optional
    :param block: hold plot until figure is closed, defaults to None
    :type block: bool, optional
    :param kwargs: arguments passed to :meth:`spatialmath.base.graphics.plot_ellipse`

    Plot ``N`` uncertainty ellipses spaced evenly along the trajectory.

    :seealso: :meth:`get_P` :meth:`run` :meth:`history`
    """
    nhist = len(self._history)

    if "label" in kwargs:
        label = kwargs["label"]
        del kwargs["label"]
    else:
        label = f"{confidence*100:.3g}% confidence"

    for k in np.linspace(0, nhist - 1, N):
        k = round(k)
        h = self._history[k]
        if k == 0:
            base.plot_ellipse(
                h.P[:2, :2],
                centre=h.xest[:2],
                confidence=confidence,
                label=label,
                inverted=True,
                **kwargs,
            )
        else:
            base.plot_ellipse(
                h.P[:2, :2],
                centre=h.xest[:2],
                confidence=confidence,
                inverted=True,
                **kwargs,
            )
    if block is not None:
        plt.show(block=block)

def plot_vehicle(Vehicle):
    fig, ax = plt.subplots()

    nframes = round(T / self.dt)
    anim = animation.FuncAnimation(
        fig=fig,
        func=lambda i: self.step(animate=True, pause=False),
        init_func=lambda: self.init(animate=True),
        frames=nframes,
        interval=self.dt * 1000,
        blit=False,
        repeat=False,
    )

def plot_ekf(EKF):
    fig, ax = plt.subplots()

    def init():
        self.init()
        if self.sensor is not None:
            self.sensor.map.plot()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    def animate(i):
        self.robot._animation.update(self.robot.x)
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
   
def ekf_predict():
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


def ekf_sensor(EKF):#
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


def EKF_update(ekf):#
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

    ## landmark management

    def _isseenbefore(self, lm_id):

        # _landmarks[0, id] is the order in which seen
        # _landmarks[1, id] is the occurence count

        return lm_id in self._landmarks

    def _landmark_increment(self, lm_id):
        self._landmarks[lm_id][1] += 1  # update the count

    def _landmark_count(self, lm_id):
        return self._landmarks[lm_id][1]

    def _landmark_add(self, lm_id):
        self._landmarks[lm_id] = [len(self._landmarks), 1]

    def _extend_map(self, P, xv, xm, z, lm_id):
        # this is a new landmark, we haven't seen it before
        # estimate position of landmark in the world based on
        # noisy sensor reading and current vehicle pose

        # M = None

        # estimate its position based on observation and vehicle state
        xf = self.sensor.g(xv, z)

        # append this estimate to the state vector
        if self._est_vehicle:
            x_ext = np.r_[xv, xm, xf]
        else:
            x_ext = np.r_[xm, xf]

        # get the Jacobian for the new landmark
        Gz = self.sensor.Gz(xv, z)

        # extend the covariance matrix
        n = len(self._x_est)
        if self._est_vehicle:
            # estimating vehicle state
            Gx = self.sensor.Gx(xv, z)
            # fmt: off
            Yz = np.block([
                [np.eye(n), np.zeros((n, 2))    ],
                [Gx,        np.zeros((2, n-3)), Gz]
            ])
            # fmt: on
        else:
            # estimating landmarks only
            # P_ext = block_diag(P, Gz @ self._W_est @ Gz.T)
            # fmt: off
            Yz = np.block([
                [np.eye(n),        np.zeros((n, 2))    ],
                [np.zeros((2, n)), Gz]
            ])
            # fmt: on
        P_ext = Yz @ block_diag(P, self._W_est) @ Yz.T

        return x_ext, P_ext

if __name__=="__main__":
  

  bk = Bike()
  bk.init_state(0., 0., np.pi * 0.1)
  



