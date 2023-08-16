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

def convert_experiment_to_dict() -> dict:
    """
        Function to convert an experiment with sensors, robots and things into a dictionary for yaml storage
    """
    raise NotImplementedError("Not implemented yet. get to it")

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
    with open(fpath, 'w') as outfile:
        yaml.dump(somedict, outfile, default_flow_style=False)
    print("Written yaml to: {}".format(fpath))

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