import numpy as np
import os
import pickle
from datetime import datetime
import itertools
import logging
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def get_logger():
    log = logging.getLogger("ssl")
    log.setLevel(logging.DEBUG)

    if not log.hasHandlers():
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        log.addHandler(handler)

    return log


def flatten_list(my_list):
    return list(itertools.chain.from_iterable(my_list))


def save_dict(dict, savefile):
    """Save dict file as .pkl"""
    file_extension = savefile.rsplit(".", 1)[-1].lower()

    if file_extension == "pkl":
        dirname = os.path.dirname(savefile)

        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(savefile, "wb") as file:
            pickle.dump(dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        raise ValueError("Unsupported file type. Only '.pkl' is supported.")


def load_dict(savefile):
    """Load .pkl file as dict"""
    file_extension = savefile.rsplit(".", 1)[-1].lower()

    if file_extension == "pkl":
        with open(savefile, "rb") as file:
            return pickle.load(file)

    else:
        raise ValueError("Unsupported file type. Only '.pkl' is supported.")


def check_files_exist(dir, files):
    """Ensure all expected files exist in dir"""
    return all(os.path.exists(os.path.join(dir, file)) for file in files)


def ordered_unique(x):
    """Return unique elements, maintaining order of appearance"""
    return x[np.sort(np.unique(x, return_index=True)[1])]


def get_first_file(dataFolder, folderName):
    return os.path.join(
        dataFolder, folderName, os.listdir(os.path.join(dataFolder, folderName))[0]
    )


def load_environment_vars(env_strings=[]):
    missing_envs = []
    env_values = []

    for env_string in env_strings:
        env_value = os.getenv(env_string)
        if env_value is None or env_value == "":
            missing_envs.append(env_string)
        else:
            env_values.append(env_value)

    if missing_envs:
        missing_envs_str = ", ".join(missing_envs)
        raise ValueError(
            f"Please set the following environment variable(s) in the .env file: {missing_envs_str}"
        )

    return tuple(env_values)


def parse_datetime_from_timestamp(time_in_secs):
    return datetime.fromtimestamp(float(time_in_secs))


class dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = dotdict(value)
            self[key] = value
