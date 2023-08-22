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
import yaml
import json

"""
    ToDO: write conversion file to yaml:
    https://stackoverflow.com/questions/65820633/dumping-custom-class-objects-to-a-yaml-file
    may be too complicated - just turn into a dict.
    use function below
    use a __dict__ function in classes that we implement ourselves
    use vars(sensor)?
"""

def dump_json(somedict : dict, fpath):
    with open(fpath, "w") as outf:
        json.dump(somedict, outf, cls=NumpyEncoder)
    return None

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

def to_yaml(somedict : dict, dirname : str, fname : str) -> None:
    """
        Function to write dictionary to directory with filename
    """
    fpath = os.path.join(dirname, fname)
    if os.path.isfile(fpath):
        print("{} already exists. Appending date for unique filenames")
        fpath = _change_filename(fpath)
    if not os.path.exists(dirname):
        _create_dir(dirname)
    fpath = fpath + ".yml"
    sd = convert_experiment_to_dict(somedict)
    with open(fpath, 'w') as outfile:
        yaml.dump(sd, outfile, default_flow_style=False)
    print("Written yaml to: {}".format(fpath))

def load_yaml(fpath : str):
    with open(fpath, 'r') as inf:
        ind = yaml.load(inf, Loader=yaml.Loader)
    return ind

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
        json.dump(outdict, outfile, cls=NumpyEncoder)
    print("Written {} to {}".format(hname, outf))

def _create_dir(dirname : str) -> None:
    """
        Function to create a directory in a parentdir
    """
    import os
    os.makedirs(dirname)

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