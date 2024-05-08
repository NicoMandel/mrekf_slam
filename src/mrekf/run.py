"""
    basic example using inherited functions.
    Todo in order of priority
    1. check the normal simulation run works - no errors
    2. adapt sensor model to work with dictionary
    3. check histories can be used to create evaluations 
    4. include logging module for debugging purposes
"""

import numpy as np
from omegaconf import DictConfig
from roboticstoolbox import RandomPath

# own import
from mrekf.simulation import Simulation
from mrekf.utils import convert_simulation_to_dict
from mrekf.debug_utils import _check_history_consistency, _compare_filter_and_new
from mrekf.init_params import init_robot, init_map, init_motion_model, init_dyn, init_filters, init_simulation_sensor


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
        fp_count = len(cfg.fp_list),            # todo - change this to get the list of FP ids 
        dynamic_count = len(sec_robots)
    )

    # basedir = os.path.abspath(os.path.join(os.path.dirname(__file__) , '..', '..'))
    # videofpath = os.path.join(basedir, 'results', 'tmp.mp4')
    # html = sim.run_animation(T=time, format="mp4", file=videofpath) # format=None 
    sim.run_simulation(T=time)

    if cfg.debug:
        _check_history_consistency(sim.history, ekf_list)
        ekf_histd = {ekf.description : ekf.history for ekf in ekf_list}
        _compare_filter_and_new(ekf_histdict=ekf_histd, cfg=cfg, gt_hist=sim.history)
    # plt.show()
    
    hists = {ekf.description : ekf.history for ekf in ekf_list}

    return simdict, sim.history, hists    