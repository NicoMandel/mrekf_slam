"""
    basic example using inherited functions.
    Todo in order of priority
    1. check the normal simulation run works - no errors
    2. adapt sensor model to work with dictionary
    3. check histories can be used to create evaluations 
    4. include logging module for debugging purposes
"""

import numpy as np
import os.path
# np.set_printoptions(precision=4, suppress=True, linewidth=10000, edgeitems=30)
from math import pi
import matplotlib.pyplot as plt

from roboticstoolbox import LandmarkMap, Bicycle, RandomPath

# own import
from mrekf.simulation import Simulation
from mrekf.ekf_base import BasicEKF
from mrekf.dynamic_ekf import Dynamic_EKF
from mrekf.sensor import  RobotSensor, get_sensor_model
from mrekf.motionmodels import BaseModel, StaticModel, KinematicModel, BodyFrame
from mrekf.utils import convert_simulation_to_dict


def init_robot(configs : dict) -> tuple[Bicycle, np.ndarray]:
    """
        Function to initialize the robot
    """
    Vr = configs['vehicle_model']['V']
    Vr[1] = np.deg2rad(Vr[1])
    V_r1 = np.diag(Vr) ** 2
    x0r = configs["vehicle_model"]['x0']
    x0r[2] = np.deg2rad(x0r[2])
    # rtype = configs["vehicle_model"]["type"]
    robot = Bicycle(covar=V_r1, x0=x0r, 
            animation="car")
    return robot, V_r1

def init_sensor(configs : dict, mot_model : BaseModel, robot : Bicycle, lm_map : LandmarkMap, sec_robots : dict) -> tuple[RobotSensor, np.ndarray]:
    """
        Function to initialize the sensor
    """
    rg = configs["sensor"]["range"]
    ang = configs["sensor"]["angle"]
    ang = pi if ang is None else ang
    # W = np.diag([0.4, np.deg2rad(10)]) ** 2
    W_mod = configs["sensor"]["W"]
    W_mod[1] = np.deg2rad(W_mod[1])
    W = np.diag(W_mod) ** 2
    sensor = get_sensor_model(mot_model, robot=robot, lm_map=lm_map,
                               r2=sec_robots, covar= W, 
                               rg = rg, angle=[-ang, ang])
    return sensor, W

def init_map(experiment : dict) -> LandmarkMap:
    """
        Function to initialize the map
    """
    s_lms = experiment["static"]
    ws = experiment["workspace"]
    lm_map = LandmarkMap(s_lms, workspace=ws)
    return lm_map

def init_dyn(experiment : dict, configs : dict, lm_map : LandmarkMap) -> dict:
    """
        Function to initialize the dynamic landmarks
    """
    d_lms = experiment["dynamic"]
    robot_offset = experiment["offset"]
    
    Vr = configs['vehicle_model']['V']
    Vr[1] = np.deg2rad(Vr[1])
    V_r1 = np.diag(Vr) ** 2

    sec_robots = {}
    for i in range(d_lms):
        # V_r2 = np.diag([0.2, np.deg2rad(5)]) ** 2
        V_r2 = V_r1.copy()
        r2 = Bicycle(covar=V_r2, x0=(np.random.randint(-10, 10), np.random.randint(-10, 10), np.deg2rad(np.random.randint(0,360))), animation="car")
        r2.control = RandomPath(workspace=lm_map, seed=None)
        r2.init()
        sec_robots[i + robot_offset] = r2
    return sec_robots

def init_motion_model( configs : dict, dt : float = None) -> BaseModel:
    
    mmtype : str = configs["motion_model"]["type"]
    V_mm : np.ndarray = configs["motion_model"]["V"]
    if "body" in mmtype.lower():
        V_mm[1] = np.deg2rad(V_mm[1])
        V_est = np.diag(V_mm) ** 2
        mot_model = BodyFrame(V_est, dt=dt)
    elif "kinematic" in mmtype.lower():
        V_est = np.diag(V_mm) ** 2
        mot_model = KinematicModel(V_est, dt)
    elif "static" in mmtype.lower():
        V_est = np.diag(V_mm) ** 2
        mot_model = StaticModel(V_est)
    else:
        raise NotImplementedError("Unknown Motion Model of Type: {}. Known are BodyFrame, Kinematic or Static, see motion_models file".format(mmtype))
    return mot_model

def init_filters(experiment : dict, configs : dict, robot_est : tuple[Bicycle, np.ndarray], sensor_est : tuple[RobotSensor, np.ndarray], mot_model : BaseModel, sec_robots : dict, history : bool) -> list[BasicEKF]:
    """
        Function to initialize the filters
    """

    x0_est_raw = configs["init"]["x0"]
    x0_est = np.array(x0_est_raw)
    P0_est_raw = configs["init"]["P0"]
    P0_est_raw[2] = np.deg2rad(P0_est_raw[2])
    P0 = np.diag(P0_est_raw) ** 2

    ekf_list = []
    
    # excluding -> basic ekf
    x0_exc = x0_est.copy()
    P0_exc = P0.copy()
    ekf_exc = BasicEKF(
        description="EKF_EXC",
        x0=x0_exc, P0=P0_exc, robot=robot_est, sensor=sensor_est,
        ignore_ids=list(sec_robots.keys()),
        history=history
    )
    ekf_list.append(ekf_exc)

    # including -> basic ekf
    if experiment["incfilter"]:
        x0_inc = x0_est.copy()    
        P0_inc = P0.copy()
        ekf_inc = BasicEKF(
            description="EKF_INC",
            x0=x0_inc, P0=P0_inc, robot=robot_est, sensor=sensor_est,
            ignore_ids=[],
            history=history
        )
        ekf_list.append(ekf_inc)
        
    # Dynamic EKFs
    # FP -> dynamic Ekf    
    if experiment["fpfilter"]:
        x0_fp = x0_est.copy()
        P0_fp = P0.copy()
        fp_list = configs["fp_list"]
        ekf_fp = Dynamic_EKF(
            description="EKF_FP",
            x0=x0_fp, P0=P0_fp, robot=robot_est, sensor = sensor_est,
            motion_model=mot_model,
            dynamic_ids=fp_list,
            ignore_ids=list(sec_robots.keys()),
            history=history
        )
        ekf_list.append(ekf_fp)

    # real one
    if experiment["dynamicfilter"]:
        ekf_mr = Dynamic_EKF(
            description="EKF_MR",
            x0=x0_est, P0=P0, robot=robot_est, sensor=sensor_est,
            motion_model=mot_model,
            dynamic_ids=list(sec_robots.keys()),
            history=history
        )
        ekf_list.append(ekf_mr)
        
    return ekf_list

def run_simulation(experiment : dict, configs: dict) -> tuple[dict, dict, dict]:
    """
        Function to run the simulation from a dictionary containing the keywords and another containing the configurations.
    """
    # general experimental setting
    history=True
    verbose=True                # todo use this to set the loglevel
    seed = experiment["seed"]
    np.random.seed(seed)
    time = experiment["time"]

    # Setup robot 1
    robot, V = init_robot(configs)
    
    # setup map - used for workspace config
    lm_map = init_map(experiment)
    robot.control = RandomPath(workspace=lm_map, seed=seed)
    
	# Setup secondary Robots  
    sec_robots = init_dyn(experiment, configs, lm_map)
    
    # Setup estimate functions for the second robot. the sensor depends on this!
    mot_model = init_motion_model(configs, robot.dt)
    
    # Setup Sensor
    sensor, W = init_sensor(configs, mot_model, robot, lm_map, sec_robots)

    ##########################
    # EKF SETUPS
    ##########################
    P0_est_raw = configs["init"]["P0"]
    P0_est_raw[2] = np.deg2rad(P0_est_raw[2])
    P0 = np.diag(P0_est_raw) ** 2

    ekf_list = init_filters(experiment, configs, (robot, V), (sensor, W), mot_model, sec_robots, history)

    ###########################
    # RUN
    ###########################
    sim = Simulation(
        robot=(robot, V),
        r2=sec_robots,
        P0=P0,      # not used, only for inheritance. Could theoretically be a dummy value?
        sensor=(sensor, W),
        verbose=verbose,
        history=history,
        ekfs=ekf_list
    )
    simdict = convert_simulation_to_dict(sim, mot_model, seed=seed)
    
    # basedir = os.path.abspath(os.path.join(os.path.dirname(__file__) , '..', '..'))
    # videofpath = os.path.join(basedir, 'results', 'tmp.mp4')
    # html = sim.run_animation(T=time, format="mp4", file=videofpath) # format=None 
    sim.run_simulation(T=time)
    # plt.show()
    hists = {ekf.description : ekf.history for ekf in ekf_list}

    return simdict, sim.history, hists    