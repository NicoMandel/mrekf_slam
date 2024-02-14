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
from math import pi
import matplotlib.pyplot as plt
from copy import deepcopy

from roboticstoolbox import LandmarkMap, Bicycle, RandomPath

# own import
from mrekf.simulation import Simulation
from mrekf.ekf_base import BasicEKF
from mrekf.dynamic_ekf import Dynamic_EKF
from mrekf.sensor import SimulationSensor, SensorModel
from mrekf.motionmodels import BaseModel, StaticModel, KinematicModel, BodyFrame
from mrekf.utils import convert_simulation_to_dict


def init_robot(configs : dict) -> tuple[Bicycle, np.ndarray]:
    """
        Function to initialize the robot
    """
    Vr = deepcopy(configs['vehicle_model']['V'])
    Vr[1] = np.deg2rad(Vr[1])
    V_r1 = np.diag(Vr) ** 2
    x0r = deepcopy(configs["vehicle_model"]['x0'])
    x0r[2] = np.deg2rad(x0r[2])
    # rtype = configs["vehicle_model"]["type"]
    robot = Bicycle(covar=V_r1, x0=x0r, 
            animation="car")
    return robot, V_r1

def _sensor_from_configs(configs : dict) -> tuple:
    rg = configs["sensor"]["range"]
    ang = configs["sensor"]["angle"]
    ang = pi if not ang else ang
    # W = np.diag([0.4, np.deg2rad(10)]) ** 2
    W_mod = deepcopy(configs["sensor"]["W"])
    W_mod[1] = np.deg2rad(W_mod[1])
    W = np.diag(W_mod) ** 2
    return rg, ang, W

def init_simulation_sensor(configs : dict, robot : Bicycle, lm_map : LandmarkMap, sec_robots : dict, robot_offset : int = 100) -> tuple[SimulationSensor, np.ndarray]:
    """
        Function to initialize the sensor
    """
    rg, angle, W = _sensor_from_configs(configs)
    sensor = SimulationSensor(
        robot=robot,
        r2=sec_robots,
        lm_map=lm_map,
        angle=angle,
        robot_offset=robot_offset,
        range=rg,
        covar=W,
    )
    return sensor, W

def init_sensor_model(configs : dict, robot : Bicycle, lm_map : LandmarkMap) -> tuple[SensorModel, np.ndarray]:
    """
        Function to initialize the sensor model for each of the ekfs
    """
    _, _, W = _sensor_from_configs(configs)
    sensor = SensorModel(
        robot=robot,
        lm_map=lm_map,
        # covar=W,
    )
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
    
    Vr = deepcopy(configs['vehicle_model']['V'])
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

def init_motion_model(configs : dict, dt : float = None) -> list[BaseModel]:
    """
        Returns a list of motion models - one for each configured in configs.
    """
    mmls = configs["motion_model"]
    if isinstance(mmls, list):
        mml = [_init_motion_model(mm["type"], mm["V"], dt) for mm in mmls]
    else:
        mml = [_init_motion_model(mmls["type"], mmls["V"], dt)]    
    return mml

def _init_motion_model( mmtype: str, V : np.ndarray, dt : float = None) -> list[BaseModel]:
    """
        Returns a list of models - one for each configured in the configs
    """
    V_mm = deepcopy(V)
    if "body" in mmtype.lower():
        V_mm[1] = np.deg2rad(V_mm[1])
        V_est = np.diag(V_mm) ** 2
        mot_model = BodyFrame(V_est, dt=dt)
    elif "kinematic" in mmtype.lower():
        V_est = np.diag(V_mm) ** 2
        mot_model = KinematicModel(V_est, dt=dt)
    elif "static" in mmtype.lower():
        V_est = np.diag(V_mm) ** 2
        mot_model = StaticModel(V_est, dt=dt)
    else:
        raise NotImplementedError("Unknown Motion Model of Type: {}. Known are BodyFrame, Kinematic or Static, see motion_models file".format(mmtype))
    return mot_model

def init_filters(experiment : dict, configs : dict, robot_est : tuple[Bicycle, np.ndarray], lm_map : LandmarkMap, mot_models : list[BaseModel], sec_robots : dict, history : bool) -> list[BasicEKF]:
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
    x0_exc = deepcopy(x0_est)
    P0_exc = deepcopy(P0)
    sensor_exc = init_sensor_model(configs, robot_est[0], lm_map)
    ekf_exc = BasicEKF(
        description="EKF_EXC",
        x0=x0_exc, P0=P0_exc, robot=robot_est, sensor=sensor_exc,
        ignore_ids=list(sec_robots.keys()),
        history=history
    )
    ekf_list.append(ekf_exc)

    # including -> basic ekf
    if experiment["incfilter"]:
        x0_inc = deepcopy(x0_est)    
        P0_inc = deepcopy(P0)
        sensor_inc = init_sensor_model(configs, robot_est[0], lm_map)
        ekf_inc = BasicEKF(
            description="EKF_INC",
            x0=x0_inc, P0=P0_inc, robot=robot_est, sensor=sensor_inc,
            ignore_ids=[],
            history=history
        )
        ekf_list.append(ekf_inc)
        
    # Dynamic EKFs
    # FP -> dynamic Ekf    
    if experiment["fpfilter"]:
        fp_list = configs["fp_list"]
        for mm in mot_models:
            x0_fp = deepcopy(x0_est)
            P0_fp = deepcopy(P0)
            sensor_fp = init_sensor_model(configs, robot_est[0], lm_map)
            ekf_fp = Dynamic_EKF(
                description="EKF_FP:{}".format(mm.abbreviation),
                x0=x0_fp, P0=P0_fp, robot=robot_est, sensor = sensor_fp,
                motion_model=mm,
                dynamic_ids=fp_list,
                ignore_ids=list(sec_robots.keys()),
                history=history
            )
            ekf_list.append(ekf_fp)

    # real one
    if experiment["dynamicfilter"]:
        for mm in mot_models:
            x0_mr = deepcopy(x0_est)
            P0_mr = deepcopy(P0)
            sensor_mr = init_sensor_model(configs, robot_est[0], lm_map)
            ekf_mr = Dynamic_EKF(
                description="EKF_MR:{}".format(mm.abbreviation),
                x0=x0_mr, P0=P0_mr, robot=robot_est, sensor=sensor_mr,
                motion_model=mm,
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
    robot_offset =  experiment["offset"]

    # Setup robot 1
    robot, V = init_robot(
        configs=configs
        )
    
    # setup map - used for workspace config
    lm_map = init_map(
        experiment=experiment
        )
    robot.control = RandomPath(workspace=lm_map, seed=seed)
    
	# Setup secondary Robots  
    sec_robots = init_dyn(
        experiment=experiment,
        configs=configs,
        lm_map=lm_map
        )
    
    # Setup estimate functions for the second robot. the sensor depends on this!
    mot_models = init_motion_model(
        configs=configs,
        dt=robot.dt
        )
    
    # Setup Simulation Sensor
    sensor, W = init_simulation_sensor(
        configs=configs,
        robot=robot,
        lm_map=lm_map,
        sec_robots=sec_robots,
        robot_offset=robot_offset
        )
    # init_sensor(configs, mot_models, robot, lm_map, sec_robots)

    ##########################
    # EKF SETUPS
    ##########################
    P0_est_raw = configs["init"]["P0"]
    P0_est_raw[2] = np.deg2rad(P0_est_raw[2])
    P0 = np.diag(P0_est_raw) ** 2

    ekf_list = init_filters(
        experiment=experiment,
        configs=configs,
        lm_map=lm_map,
        robot_est=(robot, V),
        mot_models=mot_models,
        sec_robots=sec_robots,
        history=history
        )

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
    simdict = convert_simulation_to_dict(
        sim=sim,
        mot_models=mot_models,
        seed=seed
        )
    
    # basedir = os.path.abspath(os.path.join(os.path.dirname(__file__) , '..', '..'))
    # videofpath = os.path.join(basedir, 'results', 'tmp.mp4')
    # html = sim.run_animation(T=time, format="mp4", file=videofpath) # format=None 
    sim.run_simulation(T=time)
    # plt.show()
    hists = {ekf.description : ekf.history for ekf in ekf_list}

    return simdict, sim.history, hists    