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
import base64
from typing import Any
import numpy as np
import os.path
from datetime import date, datetime
import json
import h5py
import pickle

# Section on dumping files
def dump_json(somedict : dict, fpath):
    with open(fpath, "w") as outf:
        json.dump(somedict, outf, cls=NumpyEncoder)
    return None

def dump_json_nice(somedict : dict, fpath):
    with open(fpath, "w") as outf:
        json.dump(somedict, outf, cls=NumpyEncoderNice)
    return None

def load_json_nice(fpath) -> dict:
    with open(fpath, 'r') as f:
        res = json.load(f, object_hook=json_np_obj_hook)
    return res


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

    sd['robot'] = get_robot_values(r)
    sd['sensor']  = get_sensor_values(sens)
    sd['map']  = get_map_values(lm_map)
    sd['model'] = get_mot_model_values(motion_model)
    sd['robots'] = get_robs_values(robots, sens.robot_offset if sens.robot_offset else 0)
    
    # sd = {k : vars(obj) if hasattr(obj, "__dict__") else obj for k, obj in somedict.items()}

    return sd

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
    lmd['landmarks'] = lm_map.landmarks.T
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

def dump_namedtuple(nt : list, dirname : str) -> None:
    """
        Function to dump list of namedtuples to json
    """
    # first step is to convert to dictionary -> by timestamp
    outdict = {h.t : h._asdict() for h in nt}
    
    # the name is the name of the history
    hname = type(nt[0]).__name__

    if not os.path.isdir(dirname):
        print("{} does not exist. Creating".format(dirname))
        _create_dir(dirname)
    
    # save the dictionary
    outf = os.path.join(dirname, hname + ".json")
    with open(outf, "w") as outfile:
        json.dump(outdict, outfile, cls=NumpyEncoderNice)
    print("Written {} to {}".format(hname, outf))

def _create_dir(dirname : str) -> None:
    """
        Function to create a directory in a parentdir
    """
    import os
    os.makedirs(dirname)

# Section on Loading arrays
def load_experiment(json_path : str) -> dict:
    """
        Function to load experiments from json
    """
    with open(json_path, 'r') as jsf:
        jsd = json.load(jsf, object_hook=json_list_np_obj_hook)
    return jsd

def load_history(hist_path : str) -> dict:
    with open(hist_path, 'r') as hf:
        hd = json.load(hf)
    # todo - data cleaning
    return hd

def list_to_numpy(dct : dict) -> dict:
    nd = {}
    for k, v in dct.items():
        if isinstance(v, dict):
            list_to_numpy(v)
        if isinstance(v, list):
            v = np.asarray(v)
        nd[k] = v
    return nd

# Jsonify numpy arrays
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
    """
        restoring arrays - needs prior knowledge of what was array! - see https://stackoverflow.com/a/47626762/8888097
        -> store as additional hidden config file?
    """

class NumpyEncoderNice(json.JSONEncoder):

    def default(self, o: Any) -> Any:
        """
            Input objects of ndarray will be converted to dict with dtype, shape and data in base64
        """
        if isinstance(o, np.ndarray):
            obj_data = np.ascontiguousarray(o).data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64, dtype=str(o.dtype), shape=o.shape)
        
        super().default(o)

def json_list_np_obj_hook(dct):
    if isinstance(dct, list):
        dct = np.asarray(dct)
    return dct

def json_np_obj_hook(dct):
    """
        Decodes np array - json combo encoded through previous thing
    """
    if isinstance(dct, dict) and "__ndarray__" in dct:
        data = base64.b64decode(dct["__ndarray__"])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct


# working with hdf5s!
def dump_h5(nt : list, dirname : str) -> None:
    """
        similar to the function for dumping namedtuples below
    """
    outdict = {h.t : h._asdict() for h in nt}
    
    # the name is the name of the history
    hname = type(nt[0]).__name__

    if not os.path.isdir(dirname):
        print("{} does not exist. Creating".format(dirname))
        _create_dir(dirname)
    
    # save the dictionary
    outf = os.path.join(dirname, hname + ".hdf5")
    with h5py.File(outf, "w") as outfile:
        json.dump(outdict, outfile, cls=NumpyEncoderNice)
    print("Written {} to {}".format(hname, outf))

def dump_pickle(nt : list, dirname : str) -> None:
    outdict = {h.t : h._asdict() for h in nt}
    
    # the name is the name of the history
    hname = type(nt[0]).__name__

    if not os.path.isdir(dirname):
        print("{} does not exist. Creating".format(dirname))
        _create_dir(dirname)
    
    # save the dictionary
    outf = os.path.join(dirname, hname + ".pkl")
    with open(outf, "wb") as outfile:
        pickle.dump(outdict, outfile)
    print("Written {} to {}".format(hname, outf))

def load_pickle(fp : str):
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    return data