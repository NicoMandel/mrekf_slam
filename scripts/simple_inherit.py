"""
    basic example using inherited functions.
    Todo in order of priority
    1. check the normal simulation run works - no errors
    2. adapt sensor model to work with dictionary
    3. check histories can be used to create evaluations 
    4. include logging module for debugging purposes
"""
import os.path
import numpy as np
from datetime import date, datetime
import matplotlib.pyplot as plt

from roboticstoolbox import LandmarkMap, Bicycle, RandomPath
from math import pi

# own import
from mrekf.simulation import Simulation
from mrekf.ekf_base import BasicEKF
from mrekf.dynamic_ekf import Dynamic_EKF
from mrekf.sensor import  RobotSensor, get_sensor_model
from mrekf.motionmodels import StaticModel, KinematicModel, BodyFrame
from mrekf.utils import convert_simulation_to_dict, dump_json, dump_pickle

if __name__=="__main__":
    # general experimental setting
    history=True
    verbose=True                # todo use this to set the loglevel
    seed = 0
    # Setup robot 1
    V_r1 = np.diag([0.2, np.deg2rad(5)]) ** 2
    robot = Bicycle(covar=V_r1, x0=(0, 0, np.deg2rad(0.1)), 
            animation="car")
    
    # setup map - used for workspace config
    lm_map = LandmarkMap(20, workspace=10)
    robot.control = RandomPath(workspace=lm_map, seed=seed)
    
    # Setup Sensor
    W = np.diag([0.2, np.deg2rad(3)]) ** 2
    # sensor = RangeBearingSensor(robot=robot, map=map, covar=W
            # range=4, angle=[-pi/2, pi/2])
	# Setup secondary Robots    
    # additional_marker= VehicleMarker()
    sec_robots = {}
    robot_offset = 100
    for i in range(1):
        V_r2 = np.diag([0.2, np.deg2rad(5)]) ** 2
        r2 = Bicycle(covar=V_r2, x0=(1, 4, np.deg2rad(45)), animation="car")
        r2.control = RandomPath(workspace=lm_map, seed=robot.control._seed+1)
        r2.init()
        sec_robots[i + robot_offset] = r2
    # robots = [r2]
    rg = 10

    # Setup state estimate - is only robot 1!
    x0_est =  np.array([0., 0., 0.])      # initial estimate
    P0 = np.diag([0.05, 0.05, np.deg2rad(0.5)]) ** 2

    # Estimate the second robot
    V_est = np.diag([0.3, 0.3]) ** 2
    V_est_kin = np.zeros((4,4))
    V_est_kin[2:, 2:] = V_est
    # mot_model = StaticModel(V_est)
    # mot_model = KinematicModel(V=V_est_kin, dt=robot.dt)
    V_est_bf = V_est_kin.copy()
    mot_model = BodyFrame(V_est_bf, dt=robot.dt)
    sensor2 = get_sensor_model(mot_model, robot=robot, r2=sec_robots, covar= W, lm_map=lm_map, rng = rg, angle=[-pi/2, pi/2])

    ##########################
    # EKF SETUPS
    ##########################
    # excluding -> base ekf
    x0_exc = x0_est.copy()
    P0_exc = P0.copy()
    ekf_exc = BasicEKF(
        description="EKF_EXC",
        x0=x0_exc, P0=P0_exc, robot=(robot, V_r1), sensor=(sensor2, W),
        ignore_ids=list(sec_robots.keys()),
        history=history
    )

    # including -> base ekf
    x0_inc = x0_est.copy()    
    P0_inc = P0.copy()
    ekf_inc = BasicEKF(
        description="EKF_INC",
        x0=x0_inc, P0=P0_inc, robot=(robot, V_r1), sensor=(sensor2, W),
        ignore_ids=[],
        history=history
    )
    
    # Dynamic EKFs
    # FP -> dynamic Ekf    
    x0_fp = x0_est.copy()
    P0_fp = P0.copy()
    fp_list = [2]
    ekf_fp = Dynamic_EKF(
        description="EKF_FP",
        x0=x0_fp, P0=P0_fp, robot=(robot, V_r1), sensor = (sensor2, W),
        motion_model=mot_model, dynamic_ids=fp_list, ignore_ids=list(sec_robots.keys()),
        history=history
    )

    # real one
    ekf_mr = Dynamic_EKF(
        description="EKF_MR",
        x0=x0_est, P0=P0, robot=(robot, V_r1), sensor=(sensor2, W),
        motion_model=mot_model, dynamic_ids=list(sec_robots.keys()),
        history=history
    )
    
    ekf_list = [
        ekf_exc,
        ekf_inc,
        ekf_fp,
        ekf_mr
    ]

    ###########################
    # RUN
    ###########################
    bdir = os.path.dirname(__file__)
    pdir = os.path.abspath(os.path.join(bdir, '..'))
    rdir = os.path.join(pdir, 'results', "inherit")
    simfpath = os.path.join(rdir, 'config.json')

    sim = Simulation(
        robot=(robot, V_r1),
        r2=sec_robots,
        P0=P0,      # not used, only for inheritance
        sensor=(sensor2, W),
        verbose=verbose,
        history=history,
        ekfs=ekf_list
    )
    simdict = convert_simulation_to_dict(sim, seed=seed)
    
    videofpath = os.path.join(rdir, 'newtest.mp4')
    html = sim.run_animation(T=15, format=None) #format="mp4", file=videofpath) # format=None
    plt.show()
    # HTML(html)

    #####################
    ## SECTION ON SAVING
    ######################
    
    ############################# 
    # SAVE Experiment
    #############################
    # get a dictionary out from the simulation to store
    dump_json(simdict, simfpath)
    for ekf in ekf_list:
        dump_pickle(ekf, rdir)