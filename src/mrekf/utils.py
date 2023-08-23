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
from datetime import date, datetime
import json
import pickle
from mrekf.ekf_base import EKFLOG, MR_EKFLOG

def convert_experiment_to_dict(somedict : dict) -> dict:
    """
        Function to convert an experiment with sensors, robots and things into a dictionary for yaml storage
    """
    sd = {}
    sens = somedict['sensor']
    robots = somedict['robots']
    motion_model = somedict['motion_model']
    lm_map = somedict['map']
    r = somedict['robot']
    fps = somedict['FP']

    sd['seed'] = somedict['seed']
    sd['robot'] = get_robot_values(r)
    sd['sensor']  = get_sensor_values(sens)
    sd['map']  = get_map_values(lm_map)
    sd['model'] = get_mot_model_values(motion_model)
    sd['robots'] = get_robs_values(robots, sens.robot_offset if sens.robot_offset else 0)
    sd['FPs'] = get_fps(fps)
    # sd = {k : vars(obj) if hasattr(obj, "__dict__") else obj for k, obj in somedict.items()}

    return sd

def get_fps(fp_list : list) -> dict:
    fpd = {}
    for i, fp in enumerate(fp_list):
        fpd[i] = fp
    return fpd

def get_sensor_values(sens) -> dict:
    sensd = {}
    sensd['class'] = sens.__class__.__name__
    sensd['robot_offset'] = sens.robot_offset
    sensd['robots'] = len(sens.r2s)
    sensd['W'] = sens._W
    sensd['range'] = sens._r_range
    sensd['angle'] = sens._theta_range
    # sensd['map'] = get_map_values(sens.map)
    return sensd

def get_map_values(lm_map) -> dict:
    lmd = {}
    lmd['workspace'] = lm_map.workspace
    lmd['num_lms'] = lm_map._nlandmarks
    lmd['landmarks'] = lm_map.landmarks
    return lmd

def get_robs_values(robots : list, robot_offset : int =0) -> dict:
    robsd = {}
    for i, rob in enumerate(robots):
        rd = get_robot_values(rob)
        someid= i + robot_offset
        robsd[someid] = rd
    return robsd

def get_mot_model_values(mot_model) -> dict:
    mmd = {}
    mmd['type'] = mot_model.__class__.__name__
    mmd['dt'] = mot_model.dt
    mmd['state_length'] = mot_model.state_length
    mmd['V'] = mot_model.V
    return mmd

def get_robot_values(rob) -> dict:
    robd = {}
    robd['class'] = rob.__class__.__name__
    robd['path'] = rob.control
    robd['steer_max'] = rob.steer_max
    robd['workspace'] = rob.workspace
    robd['Noise'] = rob._V 
    robd['dt'] = rob.dt
    robd['x0'] = rob.x0
    robd['speed_max'] = rob.speed_max
    robd['accel_max'] = rob.accel_max
    robd['wheel_base'] = rob.l 
    robd['path'] = get_path_values(rob.control)
    return robd

def get_path_values(path) -> dict:
    pd = {}
    pd['name'] = "Randompath Driver Object"
    pd['workspace'] = path.workspace
    pd['dthresh'] = path._dthresh
    return pd

def _change_filename(fname : str) -> str:
    """
        Function to append the date to a string - for unique filenames
    """
    now = datetime.now()
    app = "{}_{}:{}:{}".format(date.today(), now.hour, now.minute, now.second)
    fnm = fname + app
    return fnm

def _create_dir(dirname : str) -> None:
    """
        Function to create a directory in a parentdir
    """
    import os
    os.makedirs(dirname)

# Section on Loading arrays
def dump_json(exp_dict, fpath):
    """
        function to dump the json object
    """
    with open(fpath, 'w') as jsf:
        json.dump(exp_dict, jsf, cls=NumpyEncoder)
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

def dump_pickle(nt : list, dirname : str, name="EKFlog") -> None:
    outdict = {h.t : h._asdict() for h in nt}
    
    if not os.path.isdir(dirname):
        print("{} does not exist. Creating".format(dirname))
        _create_dir(dirname)
    
    # save the dictionary
    outf = os.path.join(dirname, name + ".pkl")
    with open(outf, "wb") as outfile:
        pickle.dump(outdict, outfile)
    print("Written {} to {}".format(name, outf))

def load_pickle(fp : str, mrekf : bool = False):
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    if mrekf:
        nd = _dict_to_MREKFLOG(data)
    else:
        nd = _dict_to_EKFLOG(data)
    return nd

def _dict_to_MREKFLOG(sd : dict) -> list:
    nd = [MR_EKFLOG(**v) for v in sd.values()]
    return nd

def _dict_to_EKFLOG(sd : dict) -> list:
    nd = [EKFLOG(**v) for v in sd.values()]
    return nd