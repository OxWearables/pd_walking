import os
import pickle


def save_dict(dict, savefile):
    """ Save dict file as .pkl """
    file_extension = savefile.rsplit(".", 1)[-1].lower()

    if file_extension == "pkl":
        dirname = os.path.dirname(savefile)

        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(savefile, 'wb') as file:
            pickle.dump(dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        raise ValueError("Unsupported file type. Only '.pkl' is supported.")


def load_dict(savefile):
    """ Load .pkl file as dict """
    file_extension = savefile.rsplit(".", 1)[-1].lower()

    if file_extension == "pkl":
        with open(savefile, 'rb') as file:
            return pickle.load(file)

    else:
        raise ValueError("Unsupported file type. Only '.pkl' is supported.")


def check_files_exist(dir, files):
    """ Ensure all expected files exist in dir """
    return all(os.path.exists(os.path.join(dir, file)) for file in files)


def get_first_file(dataFolder, folderName):
    return os.path.join(dataFolder, folderName, os.listdir(os.path.join(dataFolder, folderName))[0])


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
        missing_envs_str = ', '.join(missing_envs)
        raise ValueError(f"Please set the following environment variable(s) in the .env file: {missing_envs_str}")

    return tuple(env_values)
