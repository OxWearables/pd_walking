import os
import pandas as pd
from tqdm import tqdm
from scipy import constants
from glob import glob
from joblib import Parallel, delayed

from utils import parse_datetime_from_timestamp

RAW_DIR = "Data/LDOPA_DATA/"
PROCESSED_DIR = "Data/Ldopa_Processed/"
N_JOBS = 8


def build_acc_data(datadir=RAW_DIR, processeddir=PROCESSED_DIR, n_jobs=N_JOBS):
    subjects = build_task_reference_file(datadir, processeddir)["subject_id"].unique()

    outdir = os.path.join(processeddir, 'acc_data')
    os.makedirs(outdir, exist_ok=True)

    if len(glob(os.path.join(outdir, '*.csv'))) != len(subjects):
        Parallel(n_jobs=n_jobs)(
            delayed(build_participant_acc_data)(subject, datadir, outdir)
                for subject in tqdm(subjects))
        
    else:
        print("Acceleration data already compiled...\n")


def build_task_reference_file(datadir=RAW_DIR, outdir=PROCESSED_DIR,
                              overwrite=False):
    outFile = os.path.join(outdir, 'TaskReferenceFile.csv')

    if os.path.exists(outFile) and not overwrite:
        taskRefFile = pd.read_csv(outFile, parse_dates=["timestamp_start", "timestamp_end"])
        print(f"Using saved task reference file saved at \"{outFile}\".")

    else:
        os.makedirs(outdir, exist_ok=True)

        taskScoreFile1 = os.path.join(datadir, 'TaskScoresPartI.csv')
        taskScoreFile2 = os.path.join(datadir, 'TaskScoresPartII.csv') 
        homeTaskFile = os.path.join(datadir, 'HomeTasks.csv')

        taskScore1 = pd.read_csv(taskScoreFile1, parse_dates=["timestamp_start", "timestamp_end"], 
                                 date_parser=parse_datetime_from_timestamp)
        taskScore2 = pd.read_csv(taskScoreFile2, parse_dates=["timestamp_start", "timestamp_end"], 
                                 date_parser=parse_datetime_from_timestamp)
        taskScores = pd.concat([taskScore1, taskScore2])[["subject_id", "visit", "task_code", 
                                                          "timestamp_start", "timestamp_end"]]
        visit_to_day = {1: 1, 2: 4}

        taskScores["participant_day"] = taskScores["visit"].map(visit_to_day)
        taskScores.drop("visit", axis=1, inplace=True)

        homeTasks = pd.read_csv(homeTaskFile, parse_dates=["timestamp_start", "timestamp_end"], 
                                date_parser=parse_datetime_from_timestamp)
        homeTasks = homeTasks[["subject_id", "participant_day", "task_code", 
                               "timestamp_start", "timestamp_end"]]

        taskRefFile = pd.concat([taskScores, homeTasks]).drop_duplicates().reset_index(drop=True)

        taskRefFile.to_csv(outFile)

    return taskRefFile


def build_participant_acc_data(subject, datadir, outdir):
    os.makedirs(outdir, exist_ok=True)

    accFile = os.path.join(outdir, f"{subject}.csv")
    if not os.path.exists(accFile):
        dataFiles = [pd.read_csv(build_patient_file_path(datadir, 'GENEActiv', subject, i),
                        delimiter='\t',
                        index_col = "timestamp",
                        parse_dates=True,
                        skipinitialspace=True,
                        date_parser=parse_datetime_from_timestamp) 
                    for i in range(1, 5)]
        subjectFile = pd.concat(dataFiles).dropna().drop_duplicates()
        subjectFile = subjectFile/constants.g
        subjectFile.index.name = "timestamp"
        subjectFile.rename(columns = {"GENEActiv_X": "x", "GENEActiv_Y": "y", 
                                      "GENEActiv_Z": "z", "GENEActiv_Magnitude": "mag"}, 
                           inplace = True)
        subjectFile.to_csv(accFile)
    
    else:
        print(f"Using saved subject accelerometery data at \"{accFile}\".")


def build_patient_file_path(dataFolder, device, subject_id, index):
    return os.path.join(dataFolder, device, get_patient_folder(subject_id), 
                        f"rawdata_day{index}.txt")


def get_patient_folder(subject_id):
    subject_num, subject_loc = subject_id.split("_", 1)
    if subject_loc == "BOS":
        return "patient" + subject_num
    elif subject_loc == "NYC":
        return "patient" + subject_num + "_NY"
    else:
        raise AssertionError("Invalid subject id")


def label_acc_data(datadir=RAW_DIR, processeddir=PROCESSED_DIR, n_jobs=N_JOBS):
    taskRefFile = build_task_reference_file(datadir, processeddir)
    subjects = taskRefFile["subject_id"].unique()
    
    outdir = os.path.join(PROCESSED_DIR, 'raw_labels')
    os.makedirs(outdir, exist_ok=True)
    
    if len(glob(os.path.join(outdir, '*.csv'))) != len(subjects):
        taskDictionary = build_task_dictionary(datadir, processeddir)
        accdir = os.path.join(processeddir, 'acc_data')

        Parallel(n_jobs=n_jobs)(
            delayed(label_participant_data)(subject, taskRefFile, 
                                            taskDictionary, accdir, outdir)
                for subject in tqdm(subjects))
    
    else:
        print("Label data already compiled...\n")


def build_task_dictionary(datadir=RAW_DIR, outdir=PROCESSED_DIR):
    processedDictionaryPath = os.path.join(outdir, 'TaskDictionary.csv')
        
    if os.path.exists(processedDictionaryPath):
        taskDictionary = pd.read_csv(processedDictionaryPath, index_col="task_code")
    else:
        os.makedirs(os.path.dirname(outdir), exist_ok=True)
        
        taskDictionary = pd.read_csv(os.path.join(datadir, 'TaskCodeDictionary.csv'))
        taskDictionary["is-walking"] =\
            taskDictionary["description"].apply(is_walking_given_description)
        taskDictionary.set_index("task_code", inplace=True)
        taskDictionary.to_csv(processedDictionaryPath)

    return taskDictionary


def is_walking_given_description(description):
    return 1*(("WALKING" in description.upper()) or ("STAIRS" in description.upper()))


def label_participant_data(subject, taskRefFile, taskDictionary, accdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    labelFilePath = os.path.join(outdir, f"{subject}.csv")
    accFilePath = os.path.join(accdir, f"{subject}.csv")

    if not os.path.exists(labelFilePath):
        accFile = pd.read_csv(accFilePath, index_col=[0],
                              parse_dates=True)
        
        participantTasks = taskRefFile[taskRefFile["subject_id"] == subject]

        accFile['annotation'] = -1

        for _, task in participantTasks.iterrows():
            startTime, endTime = task[["timestamp_start", "timestamp_end"]]
            mask = (accFile.index > startTime) & (accFile.index <= endTime)
            accFile.loc[mask, 'annotation'] = taskDictionary[task["task_code"]]

        walkingLabels = accFile['annotation']
        walkingLabels.to_csv(labelFilePath)

        accFile.to_csv(accFilePath)

    else:
        print(f"Using saved subject labelled accelerometery data at \"{accFilePath}\".")
