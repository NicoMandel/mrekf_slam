"""
    Functions for:
        * writing and loading histories
        * writing experiment settings
        * Plotting paths
    
        todo:
            when saving, the .yaml and the directory should have the same name!
            * save true robot path
            * save true map
"""

import numpy as np
import os.path
from pathlib import Path
from datetime import date, datetime
import json
import yaml
import pickle
import pandas as pd
from mrekf.ekf_base import EKFLOG, GT_LOG, BasicEKF

from mrekf.simulation import Simulation
from mrekf.sensor import SimulationSensor, SensorModel
from mrekf.dynamic_ekf import Dynamic_EKF
from mrekf.motionmodels import BaseModel
from roboticstoolbox.mobile.landmarkmap import LandmarkMap

# maximum threshold for json infinity parsing
MAX_THRESHOLD = 1.7e308

def convert_simulation_to_dict(sim : Simulation, mot_models : list[BaseModel], seed : int = None) -> dict:
    """
        Function to get a dictionary which can be dumped out of a simulation
    """
    sd = {}
    sd['sensor'] = get_simulation_sensor_values(sim.sensor)
    sd['map'] = get_map_values(sim.sensor.map)
    sd['dynamic'] = get_robots_values(sim.robots)
    sd['robot'] = get_robot_values(sim.robot)
    sd['motion_model'] = get_mot_models_values(mot_models)
    sd['seed'] = seed

    for ekf in sim.ekfs:
        ekf_name = ekf.description   # todo get the name of the ekf object -> a property of the EKF
        sd[ekf_name] = get_ekf_values(ekf)      # split ekf values to check if dynamic or static

    return sd

def get_fps(fp_list : list) -> dict:
    fpd = {}
    for i, fp in enumerate(fp_list):
        fpd[i] = fp
    return fpd

def get_simulation_sensor_values(sens : SimulationSensor) -> dict:
    sensd = {}
    sensd['class'] = sens.__class__.__name__
    sensd['robot_offset'] = sens.robot_offset
    sensd['robots'] = len(sens.r2s)
    sensd['W'] = sens._W
    sensd['range'] = sens._r_range
    sensd['angle'] = sens._theta_range
    # sensd['map'] = get_map_values(sens.map)
    return sensd

def get_map_values(lm_map : LandmarkMap) -> dict:
    lmd = {}
    lmd['workspace'] = lm_map.workspace
    lmd['num_lms'] = lm_map._nlandmarks
    lmd['landmarks'] = lm_map.landmarks
    return lmd

def get_robots_values(robots : dict) -> dict:
    robsd = {}
    for k, rob in robots.items():
        rd = get_robot_values(rob)
        robsd[k] = rd
    return robsd

def get_robot_values(rob) -> dict:
    robd = {}
    robd['class'] = rob.__class__.__name__
    robd['path'] = rob.control
    robd['steer_max'] = rob.steer_max
    robd['workspace'] = rob.workspace
    robd['Noise'] = rob._V 
    robd['dt'] = rob.dt
    robd['x0'] = rob.x0
    robd['speed_max'] = rob.speed_max if not np.isinf(rob.speed_max) else MAX_THRESHOLD
    robd['accel_max'] = rob.accel_max if not np.isinf(rob.accel_max) else MAX_THRESHOLD
    robd['wheel_base'] = rob.l 
    robd['path'] = get_path_values(rob.control)
    return robd

def get_ekf_values(ekf : BasicEKF) -> dict:
    """
        Function to get experimental settings from an ekf object
    """
    if hasattr(ekf, "dynamic_ids"):
        ekfd = get_dyn_ekf_values(ekf)
    else:
        ekfd = get_stat_ekf_values(ekf)
    return ekfd

def get_dyn_ekf_values(ekf : Dynamic_EKF) -> dict:
    """
        Function to get experimental settings from a dynamic EKF object
    """
    ekfd = {}
    ekfd['motion_model'] = _get_mot_model_values(ekf.motion_model)
    ekfd['dynamic_lms'] = ekf.dynamic_ids
    ekfd['use_true_init'] = ekf.use_true
    statekfd = get_stat_ekf_values(ekf)
    ekfd.update(statekfd)
    return ekfd

def get_stat_ekf_values(ekf : BasicEKF) -> dict: 
    """
        Function to get experimental settings from a basic ekf object
    """
    ekfd = {}
    ekfd["sensor_covar"] = ekf.W_est
    ekfd["vehicle_covar"] = ekf.V_est
    ekfd["ignore_ids"] = ekf.ignore_ids
    return ekfd

def get_mot_models_values(mot_models : list[BaseModel]) -> list[dict]:
    mmdl = []
    for mot_model in mot_models:
        mmd = _get_mot_model_values(mot_model)
        mmdl.append(mmd)
    return mmdl

def _get_mot_model_values(mot_model : BaseModel) -> dict:
    mmd = {}
    mmd['type'] = mot_model.__class__.__name__
    mmd['dt'] = mot_model.dt if hasattr(mot_model, "dt") else MAX_THRESHOLD
    mmd['state_length'] = mot_model.state_length
    mmd['V'] = mot_model.V
    return mmd

def get_path_values(path) -> dict:
    pd = {}
    pd['name'] = "Randompath Driver Object"
    pd['workspace'] = path.workspace
    pd['dthresh'] = path._dthresh
    pd['entropy'] = str(path._random.bit_generator._seed_seq.entropy)           # WTF
    return pd

def _create_dir(dirname : str) -> None:
    """
        Function to create a directory in a parentdir
    """
    import os
    os.makedirs(dirname)

# Section on Loading simulation dictionaries
def dump_json(exp_dict : dict, fpath : str) -> None:
    """
        function to dump the json object
    """
    with open(fpath, 'w') as jsf:
        json.dump(exp_dict, jsf, cls=NumpyEncoder)
    print("Written json to: {}".format(fpath))
    return None

def load_json(json_path : str) -> dict:
    """
        Function to load experiments from json
    """
    with open(json_path, 'r') as jsf:
        jsd = json.load(jsf)
    return jsd

# Jsonify numpy arrays
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
    """
        restoring arrays - needs prior knowledge of what was array! - see https://stackoverflow.com/a/47626762/8888097
    """

# Section on loading histories from pickle files
def load_histories_from_dir(dirname : str) -> dict:
    """
        Function to load histories from a directory.
        loads all .pkl files with the filename
    """

    dn = Path(dirname)
    fs = dn.glob("*.pkl")
    assert fs, "Path {} is empty. Check again".format(dirname)
    hd = {f.stem : load_ekf(f) for f in fs if "GT" not in f.stem}
    return hd

def load_gt_from_dir(dirname : str):
    dn = Path(dirname)
    fs = list(dn.glob("GT*.pkl"))
    return load_gt(fs[0])
    
def dump_ekf(ekf_hist, name : str, dirname : str) -> None:
    """
        Function to dump an ekf history log
    """
    
    outdict = {h.t : h._asdict() for h in ekf_hist}
    
    if not os.path.isdir(dirname):
        print("{} does not exist. Creating".format(dirname))
        _create_dir(dirname)
    
    # save the dictionary
    outf = os.path.join(dirname, name + ".pkl")
    with open(outf, "wb") as outfile:
        pickle.dump(outdict, outfile)
    print("Written {} to {}".format(name, outf))

def dump_gt(simhist, dirname : str) -> None:
    """
        Function to dump a Ground Truth Log from a simulation object
    """
    name = "GT"
    outdict = {h.t : h._asdict() for h in simhist}
    outf = os.path.join(dirname, name + ".pkl")
    with open(outf, "wb") as outfile:
        pickle.dump(outdict, outfile)
    print("Written {} to {}".format(name, outf))


def load_ekf(fp : Path) -> list:
    """
        Function to load an EKF Log
    """
    with open(fp, 'rb') as f:
        data = pickle.load(f)

    nd = _dict_to_EKFLOG(data)
    return nd

def _dict_to_EKFLOG(sd : dict) -> list:
    nd = [EKFLOG(**v) for v in sd.values()]
    return nd

def load_gt(fp : Path) -> list:
    """
        Function to load a Ground Truth Log
    """
    with open(fp, 'rb') as f:
        data = pickle.load(f)

    nd = _dict_to_GTLOG(data)
    return nd

def _dict_to_GTLOG(sd : dict) -> list:
    nd = [GT_LOG(**v) for v in sd.values()]
    return nd

def read_config(path : str) -> dict:
    """
        Utility function to read yaml config file and return
    """
    with open(path, 'r') as f:
        rd = yaml.safe_load(f)
    return rd

# Loading a csv-file and loading the recorded values for an experiment
def load_res_csv(fpath : str) -> pd.DataFrame:
    df = pd.read_csv(fpath, index_col=0)
    return df

def load_exp_from_csv(fpath : str, exp_id : str) -> pd.Series:
    df = load_res_csv(fpath)
    return df.loc[exp_id]