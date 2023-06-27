import matplotlib.pyplot as plt

# add RTB examples folder to the path
# import sys, os.path
import RVC3 as rvc
# sys.path.append(os.path.join(rvc.__path__[0], 'models'))

# ------ standard imports ------ #
import numpy as np
import math
from math import pi
np.set_printoptions(
    linewidth=120, formatter={
        'float': lambda x: f"{0:8.4g}" if abs(x) < 1e-10 else f"{x:8.4g}"})
np.random.seed(0)
from spatialmath import *
from spatialmath.base import *
from roboticstoolbox import *

# # 6.1 Dead Reckoning using Odometry
# 

# ## 6.1.1 Modeling the Robot
# 

# Robot1
v1 = np.diag([0.02, np.deg2rad(0.5)]) ** 2
r1 = Bicycle(covar=v1, animation="car")
odo = r1.step((1, 0.3))
r1.q
r1.f([0, 0, 0], odo)
r1.control = RandomPath(workspace=10)


# Robot2
v2 = np.diag([0.02, np.deg2rad(0.5)]) ** 2
r2 = Bicycle(covar=v2, animation="triangle")
r2.control = RandomPath(workspace=10, seed=1)


# EKF
r1.Fx([0, 0, 0], [0.5, 0.1])
x_sdev = [0.05, 0.05, np.deg2rad(0.5)]
P0 = np.diag(x_sdev) ** 2
ekf = EKF(robot=(r1, v1), P0=P0)

html = ekf.run_animation(T=20, format=None)

# Plotting things
r1.plot_xy(color="b")
ekf.plot_xy(color="r")

P150 = ekf.get_P(150)

np.sqrt(P150[0, 0])

ekf.plot_xy(color="r")
ekf.plot_ellipse(filled=True, facecolor="g", alpha=0.3)


t = ekf.get_t();
pn = ekf.get_Pnorm();
plt.plot(t, pn);


# # 6.2 Localizing with a Landmark Map
# 

# In[ ]:


map = LandmarkMap(20, workspace=10)


# In[ ]:


map.plot()


# In[ ]:


W = np.diag([0.1, np.deg2rad(1)]) ** 2;


# In[ ]:


sensor = RangeBearingSensor(robot=robot, map=map, covar=W,  
           angle=[-pi/2, pi/2], range=4, animate=True)


# In[ ]:


z, i = sensor.reading()
# z
# i
print(f"landmark {i} at {z}")


# In[ ]:


map[15]


# In[ ]:


map = LandmarkMap(20, workspace=10);
V = np.diag([0.02, np.deg2rad(0.5)]) ** 2
robot = Bicycle(covar=V, animation="car");
robot.control = RandomPath(workspace=map, seed=0)
W = np.diag([0.1, np.deg2rad(1)]) ** 2
sensor = RangeBearingSensor(robot=robot, map=map, covar=W, 
           angle=[-pi/2, pi/2], range=4, seed=0, animate=True);
P0 = np.diag([0.05, 0.05, np.deg2rad(0.5)]) ** 2;
ekf = EKF(robot=(robot, V), P0=P0, map=map, sensor=(sensor, W));


# In[ ]:


# ekf.run(T=20)
html = ekf.run_animation(T=20, format=None)

# In[ ]:


map.plot()
robot.plot_xy()
ekf.plot_xy()
ekf.plot_ellipse()


# # 6.3 Creating a Landmark Map
# 

# In[ ]:


map = LandmarkMap(20, workspace=10, seed=0)
robot = Bicycle(covar=V, animation="car")
robot.control = RandomPath(workspace=map)
W = np.diag([0.1, np.deg2rad(1)]) ** 2
sensor = RangeBearingSensor(robot=robot, map=map, covar=W, 
           range=4, angle=[-pi/2, pi/2], animate=True)
ekf = EKF(robot=(robot, None), sensor=(sensor, W))


# In[ ]:


# ekf.run(T=100);

html = ekf.run_animation(T=100, format=None)


# In[ ]:


map.plot()
ekf.plot_map()
robot.plot_xy()


# In[ ]:


ekf.landmark(10)


# In[ ]:


ekf.x_est[24:26]


# In[ ]:


ekf.P_est[24:26, 24:26]


# # 6.4 Simultaneous Localization and Mapping
# 

# In[ ]:


map = LandmarkMap(20, workspace=10);
W = np.diag([0.1, np.deg2rad(1)]) ** 2 
robot = Bicycle(covar=V, x0=(3, 6, np.deg2rad(-45)), 
          animation="car");
robot.control = RandomPath(workspace=map);
W = np.diag([0.1, np.deg2rad(1)]) ** 2
sensor = RangeBearingSensor(robot=robot, map=map, covar=W, 
           range=4, angle=[-pi/2, pi/2], animate=True);
P0 = np.diag([0.05, 0.05, np.deg2rad(0.5)]) ** 2;
ekf = EKF(robot=(robot, V), P0=P0, sensor=(sensor, W));


# In[ ]:


# ekf.run(T=40);

html = ekf.run_animation(T=40, format=None)


# In[ ]:


map.plot()      # plot true map
robot.plot_xy()  # plot true path


# In[ ]:


ekf.plot_map()     # plot estimated landmark position
ekf.plot_ellipse()  # plot estimated covariance
ekf.plot_xy()     # plot estimated robot path


# In[ ]:


T = ekf.get_transform(map)


# # 6.5 Pose-Graph SLAM
# 

# NOTE. Minor changes to ensure Jupyter pretty printing of SymPy equations.  Introduced underscore to indicate subscripting, and changed `t` to `theta`
# which is printed as $\theta$.

# In[ ]:


import sympy
xi, yi, ti, xj, yj, tj = sympy.symbols("x_i y_i theta_i x_j y_j theta_j")
xm, ym, tm = sympy.symbols("x_m y_m theta_m")
xi_e = SE2(xm, ym, tm).inv() * SE2(xi, yi, ti).inv() \
     * SE2(xj, yj, tj)
fk = sympy.Matrix(sympy.simplify(xi_e.xyt()))


# In[ ]:


Ai = sympy.simplify(fk.jacobian([xi, yi, ti]))
Ai.shape


# In[ ]:


Ai


# In[ ]:


pg = PoseGraph("data/pg1.g2o")


# In[ ]:


pg.plot()


# In[ ]:


# pg.optimize(animate=True)

pg.optimize(animate=False)
pg.plot()


# In[ ]:


pg = PoseGraph("data/killian-small.toro");


# In[ ]:


pg.plot(text=False)


# In[ ]:


pg.optimize()
pg.plot(text=False)


# # 6.6 Sequential Monte-Carlo Localization
# 

# In[ ]:


map = LandmarkMap(20, workspace=10)


# In[ ]:


V = np.diag([0.02, np.deg2rad(0.5)]) ** 2
robot = Bicycle(covar=V, animation="car", workspace=map)
robot.control = RandomPath(workspace=map)


# In[ ]:


W = np.diag([0.1, np.deg2rad(1)]) ** 2;
sensor = RangeBearingSensor(robot, map, covar=W, plot=True)


# In[ ]:


R = np.diag([0.1, 0.1, np.deg2rad(1)]) ** 2


# In[ ]:


L = np.diag([0.1, 0.1])


# In[ ]:


pf = ParticleFilter(robot, sensor=sensor, R=R, L=L, nparticles=1000, animate=True)


# In[ ]:


# pf.run(T=10);

html = pf.run_animation(T=10, format=None)


# In[ ]:


map.plot()
robot.plot_xy()


# In[ ]:


pf.plot_xy()


# In[ ]:


plt.plot(pf.get_std()[:100,:])


# In[ ]:


pf.plot_pdf()


# # 6.8 Application: Lidar
# 

# ## 6.8.1 Lidar-based Odometry
# 

# In[ ]:


pg = PoseGraph("data/killian.g2o.zip", lidar=True)


# In[ ]:


[r, theta] = pg.scan(100)
r.shape
theta.shape


# In[ ]:


plt.clf()
plt.polar(theta, r)


# In[ ]:


p100 = pg.scanxy(100)
p101 = pg.scanxy(100)
p100.shape


# In[ ]:


T = pg.scanmatch(100, 101)
T.printline()


# In[ ]:


pg.time(101) - pg.time(100)


# ## 6.8.2 Lidar-based Map Building
# 

# In[ ]:


og = OccupancyGrid(workspace=[-100, 250, -100, 250], cellsize=0.1, value=np.int32(0))
pg.scanmap(og, maxrange=40)
og.plot(cmap="gray")

