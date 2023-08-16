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

from collections import namedtuple
import json

def to_yaml(somedict : dict, dirname : str, fname : str) -> None:
    """
        Function to write dictionary to directory with filename
    """
    fpath = os.path.join(dirname, fname)
    if os.path.isfile(fpath):
        print("{} already exists. Appending date for unique filenames")
        fpath = _change_filename(fpath)
    fpath = fpath + ".yml"
    with open(fpath, 'w') as outfile:
        yaml.dump(somedict, outfile, default_flow_style=True)

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
        json.dump(outdict, outfile)
        print("Written {} to {}".format(hname, outf))

def _create_dir(dirname : str) -> None:
    """
        Function to create a directory in a parentdir
    """
    import os
    os.makedirs(dirname)




    
