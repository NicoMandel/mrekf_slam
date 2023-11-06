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
np.set_printoptions(precision=4, suppress=True, linewidth=10000, edgeitems=30)

from roboticstoolbox import LandmarkMap, Bicycle, RandomPath
from math import pi

# own import
from mrekf.simulation import Simulation
from mrekf.ekf_base import BasicEKF
from mrekf.dynamic_ekf import Dynamic_EKF
from mrekf.sensor import  RobotSensor, get_sensor_model
from mrekf.motionmodels import StaticModel, KinematicModel, BodyFrame
from mrekf.utils import convert_simulation_to_dict


def run_simulation(experiment : dict, configs: dict) -> dict:
    """
        Function to run the simulation from a dictionary containing the keywords and another containing the configurations.
    """
    # general experimental setting
    history=True
    verbose=True                # todo use this to set the loglevel
    seed = experiment["seed"]
    robot_offset = experiment["offset"]
    np.random.seed(seed)

    # Setup robot 1
    V_r1 = np.diag([0.2, np.deg2rad(5)]) ** 2
    robot = Bicycle(covar=V_r1, x0=(0, 0, np.deg2rad(0.1)), 
            animation="car")
    
    # setup map - used for workspace config
    s_lms = experiment["static"]
    lm_map = LandmarkMap(s_lms, workspace=10)
    robot.control = RandomPath(workspace=lm_map, seed=seed)
    
	# Setup secondary Robots  
    d_lms = experiment["dynamic"]  
    sec_robots = {}
    for i in range(d_lms):
        V_r2 = np.diag([0.2, np.deg2rad(5)]) ** 2
        r2 = Bicycle(covar=V_r2, x0=(np.random.randint(-10, 10), np.random.randint(-10, 10), np.deg2rad(np.random.randint(0,360))), animation="car")
        r2.control = RandomPath(workspace=lm_map, seed=seed+1)
        r2.init()
        sec_robots[i + robot_offset] = r2
    
    # Setup estimate functions for the second robot. the sensor depends on this!
    V_est = V_r2.copy()             # best case - where the model is just like the real thing
    V_est_kin = np.zeros((4,4))
    V_est_kin[2:, 2:] = V_est
    # mot_model = StaticModel(V_est)
    # mot_model = KinematicModel(V=V_est_kin, dt=robot.dt)
    V_est_bf = V_est_kin.copy()
    mot_model = BodyFrame(V_est_bf, dt=robot.dt)
    
    # Setup Sensor
    rg = 50
    W = np.diag([0.4, np.deg2rad(10)]) ** 2
    sensor2 = get_sensor_model(mot_model, robot=robot, lm_map=lm_map,
                               r2=sec_robots, covar= W, 
                               rg = rg, angle=[-pi, pi])

    ##########################
    # EKF SETUPS
    ##########################
    # Setup state estimate - is only robot 1!
    x0_est =  np.array([0., 0., 0.])      # initial estimate
    P0 = np.diag([0.05, 0.05, np.deg2rad(0.5)]) ** 2
    
    # excluding -> basic ekf
    x0_exc = x0_est.copy()
    P0_exc = P0.copy()
    ekf_exc = BasicEKF(
        description="EKF_EXC",
        x0=x0_exc, P0=P0_exc, robot=(robot, V_r1), sensor=(sensor2, W),
        ignore_ids=list(sec_robots.keys()),
        history=history
    )

    # including -> basic ekf
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
    fp_list = [1]
    ekf_fp = Dynamic_EKF(
        description="EKF_FP",
        x0=x0_fp, P0=P0_fp, robot=(robot, V_r1), sensor = (sensor2, W),
        motion_model=mot_model,
        dynamic_ids=fp_list,
        ignore_ids=list(sec_robots.keys()),
        history=history
    )

    # real one
    ekf_mr = Dynamic_EKF(
        description="EKF_MR",
        x0=x0_est, P0=P0, robot=(robot, V_r1), sensor=(sensor2, W),
        motion_model=mot_model,
        dynamic_ids=list(sec_robots.keys()),
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
    rdir = os.path.join(pdir, 'results', "20231106_42_33")
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
    
    videofpath = os.path.join(rdir, 'debug.mp4')
    html = sim.run_animation(T=30, format=None) # format=None format="mp4", file=videofpath

    hists = {ekf.description : ekf.history for ekf in ekf_list}
    hists["GT"] = sim.history

    return simdict, hists
    