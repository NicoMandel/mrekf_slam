"""
    basic example using inherited functions.
    Todo in order of priority
    1. check the normal simulation run works - no errors
    2. adapt sensor model to work with dictionary
    3. check histories can be used to create evaluations 
    4. include logging module for debugging purposes
"""

import numpy as np
from math import pi
from copy import deepcopy
from omegaconf import DictConfig
from hydra.utils import instantiate
from roboticstoolbox import LandmarkMap, Bicycle, RandomPath

# own import
from mrekf.simulation import Simulation
from mrekf.ekf_base import BasicEKF
from mrekf.dynamic_ekf import Dynamic_EKF
from mrekf.datmo import DATMO
from mrekf.sensor import SimulationSensor, SensorModel
from mrekf.motionmodels import BaseModel, StaticModel, KinematicModel, BodyFrame
from mrekf.utils import convert_simulation_to_dict
from mrekf.debug_utils import _check_history_consistency


def init_robot(cfg_vm : DictConfig)-> tuple[Bicycle, np.ndarray]:
    """
        Function to initialize the robot
    """
    V_r1 = _init_V(cfg_vm)
    x0r = deepcopy(np.array(cfg_vm.x0))
    x0r[2] = np.deg2rad(x0r[2])
    robot = Bicycle(covar=V_r1, x0=x0r, animation="car")
    return robot, V_r1

def _init_V(cfg_vm : DictConfig) -> np.ndarray:
    v_v, v_theta = cfg_vm.v1, cfg_vm.v2
    V_r = np.diag([v_v, np.deg2rad(v_theta)]) ** 2
    V = deepcopy(V_r)
    return V

def _init_W(cfg_s : DictConfig) -> np.ndarray:
    W_mod = np.diag([cfg_s.w_r, np.deg2rad(cfg_s.w_b)])
    W_est = deepcopy(W_mod ** 2)
    return W_est

def _sensor_from_configs(cfg_s : DictConfig) -> tuple:
    rg = cfg_s.range
    ang = cfg_s.angle 
    ang = pi if not ang else ang
    W = _init_W(cfg_s)
    return rg, ang, W

def init_simulation_sensor(cfg : DictConfig, robot : Bicycle, lm_map : LandmarkMap, sec_robots : dict, seed : int = 0, robot_offset : int = 100) -> tuple[SimulationSensor, np.ndarray]:
    """
        Function to initialize the sensor
    """
    rg, angle, W = _sensor_from_configs(cfg.sensor)
    sensor = SimulationSensor(
        robot=robot,
        r2=sec_robots,
        lm_map=lm_map,
        angle=angle,
        robot_offset=robot_offset,
        range=rg,
        covar=W,
        seed=seed
    )
    return sensor, W

def init_sensor_model(cfg_s : DictConfig, robot : Bicycle, lm_map : LandmarkMap) -> tuple[SensorModel, np.ndarray]:
    """
        Function to initialize the sensor model for each of the ekfs
    """
    _, _, W = _sensor_from_configs(cfg_s)
    sensor = SensorModel(
        robot=robot,
        lm_map=lm_map,
        covar=W,
    )
    return sensor, W

def init_map(cfg : DictConfig) -> LandmarkMap:
    """
        Function to initialize the map
    """
    s_lms = cfg.static
    ws = cfg.workspace
    lm_map = LandmarkMap(s_lms, workspace=ws)
    return lm_map

def init_dyn(cfg : DictConfig, lm_map : LandmarkMap) -> dict:
    """
        Function to initialize the dynamic landmarks
    """
    d_lms = cfg.dynamic
    robot_offset = cfg.offset
    seed = cfg.seed
    
    V_r =_init_V(cfg.vehicle_model)

    sec_robots = {}
    for i in range(d_lms):
        V_r2 = deepcopy(V_r)
        r2 = Bicycle(covar=V_r2, x0=(np.random.randint(-10, 10), np.random.randint(-10, 10), np.deg2rad(np.random.randint(0,360))), animation="car")
        r2.control = RandomPath(workspace=lm_map, seed=seed)
        r2.init()
        sec_robots[i + robot_offset] = r2
    return sec_robots

def init_motion_model(cfg_mm : dict, dt : float = None) -> list[BaseModel]:
    """
        Returns a list of motion models - one for each configured in configs.
    """
    mml = [_init_motion_model(mv['name'], np.diag(mv['V']), dt) for mk, mv in cfg_mm.items()]
    # if isinstance(cfg_mm, list):
    #     mml = [_init_motion_model(mm["type"], mm["V"], dt) for mm in cfg_mm]
    # else:
    #     mml = [_init_motion_model(cfg_mm["type"], cfg_mm["V"], dt)]    
    return mml

def _init_motion_model(mmtype: str, V : np.ndarray, dt : float = None) -> list[BaseModel]:
    """
        Returns a list of models - one for each configured in the configs
    """
    V_mm = deepcopy(V)
    if "body" in mmtype.lower():
        V_mm[1,1] = np.deg2rad(V_mm[1,1])
        V_est = V_mm ** 2
        mot_model = BodyFrame(V_est, dt=dt)
    elif "kinematic" in mmtype.lower():
        V_est = V_mm ** 2
        mot_model = KinematicModel(V_est, dt=dt)
    elif "static" in mmtype.lower():
        V_est = V_mm ** 2
        mot_model = StaticModel(V_est, dt=dt)
    else:
        raise NotImplementedError("Unknown Motion Model of Type: {}. Known are BodyFrame, Kinematic or Static, see motion_models file".format(mmtype))
    return mot_model

def init_filters(cfg : DictConfig, robot_est : tuple[Bicycle, np.ndarray], lm_map : LandmarkMap, mot_models : list[BaseModel], sec_robots : dict) -> list[BasicEKF]:
    """
        Function to initialize the filters
    """
    history = cfg.history

    x0_est = np.array(cfg.x0)
    x0_est[2] = np.deg2rad(x0_est[2])       # init_filters only run once -> deepcopy comes later
    
    P0_est =np.array(cfg.P0)
    P0_est[2] = np.deg2rad(P0_est[2])
    P0 = np.diag(P0_est) ** 2

    ekf_list = []
    
    # excluding -> basic ekf
    x0_exc = deepcopy(x0_est)
    P0_exc = deepcopy(P0)
    sensor_exc = init_sensor_model(cfg.sensor, robot_est[0], lm_map)
    ekf_exc = BasicEKF(
        description="EKF_EXC",
        x0=x0_exc, P0=P0_exc, robot=robot_est, sensor=sensor_exc,
        ignore_ids=list(sec_robots.keys()),
        history=history
    )
    ekf_list.append(ekf_exc)

    # including -> basic ekf
    filterdict = cfg.filter
    if "inclusive" in filterdict:
        x0_inc = deepcopy(x0_est)    
        P0_inc = deepcopy(P0)
        sensor_inc = init_sensor_model(cfg.sensor, robot_est[0], lm_map)
        
        ekf_inc = BasicEKF(
            description="EKF_INC",
            x0=x0_inc, P0=P0_inc, robot=robot_est, sensor=sensor_inc,
            ignore_ids=[],
            history=history
        )
        ekf_list.append(ekf_inc)
        
    # Dynamic EKFs
    # DATMO baseline
    if "datmo" in filterdict:
        for mm in mot_models:
            x0_datmo = deepcopy(x0_est)
            P0_datmo = deepcopy(P0)
            sensor_datmo = init_sensor_model(cfg.sensor, robot_est[0], lm_map)
            ekf_datmo = DATMO(
                description="EKF_DATMO:{}".format(mm.abbreviation),
                x0=x0_datmo, P0=P0_datmo, robot=robot_est, sensor=sensor_datmo,
                motion_model=mm,
                dynamic_ids=list(sec_robots.keys()),
                use_true = True if cfg.intiating else False,
                r2s=sec_robots if cfg.intiating else {},
                history=history
            )
            ekf_list.append(ekf_datmo)

    # FP -> dynamic Ekf    
    if "false_positive" in filterdict:
        fp_list = cfg.fp_list
        for mm in mot_models:
            x0_fp = deepcopy(x0_est)
            P0_fp = deepcopy(P0)
            sensor_fp = init_sensor_model(cfg.sensor, robot_est[0], lm_map)
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
    if "dynamic" in filterdict:
        for mm in mot_models:
            x0_mr = deepcopy(x0_est)
            P0_mr = deepcopy(P0)
            sensor_mr = init_sensor_model(cfg.sensor, robot_est[0], lm_map)
            ekf_mr = Dynamic_EKF(
                description="EKF_MR:{}".format(mm.abbreviation),
                x0=x0_mr, P0=P0_mr, robot=robot_est, sensor=sensor_mr,
                motion_model=mm,
                dynamic_ids=list(sec_robots.keys()),
                history=history,
                use_true= True if cfg.intiating else False,
                r2s=sec_robots if cfg.intiating else {}
            )
            ekf_list.append(ekf_mr)
        
    return ekf_list

def run_simulation(cfg : DictConfig) -> tuple[dict, dict, dict]:
    """
        Function to run the simulation from a dictionary containing the keywords and another containing the configurations.
    """
    # general experimental setting
    seed = cfg.seed
    time = cfg.time
    
    np.random.seed(seed)
    robot_offset = cfg.offset

    # Setup robot 1
    robot, V = init_robot(cfg.vehicle_model)
    
    # setup map - used for workspace config
    lm_map = init_map(cfg)

    robot.control = RandomPath(workspace=lm_map, seed=seed)
    
	# Setup secondary Robots  
    sec_robots = init_dyn(
        cfg,
        lm_map=lm_map
        )
    
    # Setup estimate functions for the second robot. the sensor depends on this!
    mot_models = init_motion_model(
        cfg_mm=cfg.motion_model,
        dt=robot.dt
        )
    
    # Setup Simulation Sensor
    sensor, W = init_simulation_sensor(
        cfg=cfg,
        robot=robot,
        lm_map=lm_map,
        sec_robots=sec_robots,
        robot_offset=robot_offset,
        seed=seed
        )
    # init_sensor(configs, mot_models, robot, lm_map, sec_robots)

    ##########################
    # EKF SETUPS
    ##########################
    ekf_list = init_filters(
        cfg,
        lm_map=lm_map,
        robot_est=(robot, V),
        mot_models=mot_models,
        sec_robots=sec_robots,
    )

    ###########################
    # RUN
    ###########################
    P0_est_raw = np.diag(cfg.P0)
    P0_est_raw[2,2] = np.deg2rad(P0_est_raw[2,2])
    P0_dummy = P0_est_raw ** 2
    sim = Simulation(
        robot=(robot, V),
        r2=sec_robots,
        P0=P0_dummy,      # not used, only for inheritance. Could theoretically be a dummy value?
        sensor=(sensor, W),
        verbose=cfg.verbose,
        history=cfg.history,
        ekfs=ekf_list
    )
    simdict = convert_simulation_to_dict(
        sim=sim,
        mot_models=mot_models,
        seed=seed,
        time=time,
        fp_count = len(cfg.fp_list),
        dynamic_count = len(sec_robots)
    )

    # basedir = os.path.abspath(os.path.join(os.path.dirname(__file__) , '..', '..'))
    # videofpath = os.path.join(basedir, 'results', 'tmp.mp4')
    # html = sim.run_animation(T=time, format="mp4", file=videofpath) # format=None 
    sim.run_simulation(T=time)

    _check_history_consistency(sim.history, ekf_list)
    # plt.show()
    hists = {ekf.description : ekf.history for ekf in ekf_list}

    return simdict, sim.history, hists    