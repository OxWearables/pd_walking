import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import warnings
import argparse
import actipy
import re
from scipy.interpolate import interp1d
from pathlib import Path
from shutil import rmtree, copyfile
from glob import glob
import urllib.request as urllib
import zipfile
import synapseclient
import synapseutils

from features import extract_features
from process_ldopa import build_acc_data, label_acc_data
from utils import check_files_exist, get_first_file, load_environment_vars

STEP_TOL = 0.4
OXWALK_DATAFILES = 'OxWalk_Dec2022/Wrist_100Hz/P[0-9][0-9]_wrist100.csv'
LDOPA_DOWNLOADS = [
    ["UPDRSResponses", "syn20681939"],
    ["TaskScoresPartII", "syn20681938"],
    ["TaskScoresPartI", "syn20681937"],
    ["TaskCodeDictionary", "syn20681936"],
    ["SensorDataPartII", "syn20681932"],
    ["SensorDataPartI", "syn20681931"],
    ["MetadataOfPatientOnboardingDictionary", "syn20681895"],
    ["MetadataOfPatientOnboarding", "syn20681894"],
    ["MetadataOfLaboratoryVisits", "syn20681892"],
    ["HomeTasks", "syn20681035"],
]
USERNAME, APIKEY = load_environment_vars(["SYNAPSE_USERNAME", "SYNAPSE_APIKEY"])

def download_oxwalk(datadir, overwrite=False):
    """ Download and extract the OxWalk dataset """
    if overwrite or not os.path.exists(os.path.join(datadir, 'OxWalk_Dec2022.zip')):
        os.makedirs(datadir, exist_ok=True)
        url = "https://ora.ox.ac.uk/objects/uuid:19d3cb34-e2b3-4177-91b6-1bad0e0163e7" + \
              "/files/dcj82k7829"
        
        with tqdm(total=3.03e8, unit="B", unit_scale=True, unit_divisor=1024, 
                  miniters=1, ascii=True, desc="Downloading OxWalk_Dec2022.zip") as pbar:
            urllib.urlretrieve(url, filename=os.path.join(datadir, "OxWalk_Dec2022.zip"),
                               reporthook=lambda b, bsize, tsize: pbar.update(bsize))

    if overwrite or len(glob(os.path.join(datadir, OXWALK_DATAFILES))) < 39:
        with zipfile.ZipFile(os.path.join(datadir, "OxWalk_Dec2022.zip"), "r") as f:
            for member in tqdm(f.namelist(), desc="Unzipping"):
                if re.match(OXWALK_DATAFILES, member):
                    try:
                        f.extract(member, datadir)
                    except zipfile.error:
                        pass
    else:
        print(f"Using saved OxWalk data at \"{datadir}\".")


def download_ldopa(datadir, overwrite=False, n_jobs=10):
    ldopa_datadir = os.path.join(datadir, 'LDOPA_DATA')
    if overwrite or (len(glob(os.path.join(ldopa_datadir, '*.csv'))) < len(LDOPA_DOWNLOADS)):
        print("Downloading data...\n")
        syn = synapseclient.login(
            USERNAME,
            apiKey = APIKEY
        )
        os.makedirs(ldopa_datadir, exist_ok=True)
        for tableName, tableId in tqdm(LDOPA_DOWNLOADS):
            syn.tableQuery(f"select * from {tableId}", includeRowIdAndRowVersion=False, 
                           downloadLocation=os.path.join(ldopa_datadir, tableName))

        for tName, _ in tqdm(LDOPA_DOWNLOADS):
            copyfile(get_first_file(ldopa_datadir, tName), os.path.join(ldopa_datadir, f'{tName}.csv'))
            rmtree(os.path.join(ldopa_datadir, tName))

    else:
        print(f"Using saved Levodopa Reponse study dictionary data at \"{ldopa_datadir}\".")


    if len(glob(os.path.join(ldopa_datadir, 'GENEActiv', '*'))) < 28:
        synapseutils.syncFromSynapse(syn, entity='syn20681023', path=datadir)

    else:
        print(f"Using saved Levodopa Reponse study accelerometery data at \"{ldopa_datadir}\".")

    processeddir = os.path.join(datadir, 'Ldopa_Processed')
    build_acc_data(ldopa_datadir, processeddir, n_jobs)
    label_acc_data(ldopa_datadir, processeddir, n_jobs)


def load_data(datafile, sample_rate=100, index_col='timestamp', annot_type='int'):
    if '.parquet' in datafile:
        data = pd.read_parquet(datafile)
        data.dropna(inplace=True)

    else:
        data = pd.read_csv(
            datafile,
            index_col=index_col, parse_dates=[index_col],
            dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': annot_type}
        )

    data, _ = actipy.process(data, sample_rate, verbose=False)

    return data


def resize(x, length, axis=1):    
    length_orig = x.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    x = interp1d(t_orig, x, kind='linear', axis=axis, assume_sorted=True)(
        t_new
    )

    return x


def make_windows(data, winsec=30, sample_rate=100, resample_rate=30, 
                 label_type='threshold', dropna=True, verbose=False, step_tol=0.4):
    X, Y, T = [], [], []
    
    for t, w in tqdm(data.resample(f"{winsec}s", origin='start'), disable=not verbose):
    
        if len(w) < 1:
            continue
    
        t = t.to_numpy()
    
        x = w[['x', 'y', 'z']].to_numpy()
        
        annot = w['annotation']
        
        if pd.isna(annot).all():  # skip if annotation is NA
            continue
    
        if not is_good_window(x, sample_rate, winsec):  # skip if bad window
            continue
                        
        if label_type == 'threshold':
            y = 'walking' if annot.sum() >= step_tol*winsec else 'not-walking'
            
        elif label_type == 'mode':
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Unable to sort modes")
                mode_label = annot.mode(dropna=False).iloc[0]
                
                if mode_label == -1:
                    continue
                
                y = 'walking' if mode_label == 1 else 'not-walking'

        if dropna and pd.isna(y):
            continue

        X.append(x)
        Y.append(y)
        T.append(t)

    X = np.stack(X)
    Y = np.stack(Y)
    T = np.stack(T)
    
    if resample_rate != sample_rate:
        X = resize(X, int(resample_rate * winsec))

    return X, Y, T


def is_good_window(x, sample_rate, winsec):
    # Check window length is correct
    window_len = sample_rate * winsec
    if len(x) != window_len:
        return False

    # Check no nans
    if np.isnan(x).any():
        return False

    return True


def load_all_and_make_windows(datadir, outdir, n_jobs, sources=['oxwalk'], overwrite=False, **kwargs):
    """ Make windows from all available data, extract features and store locally """
    if not overwrite and check_files_exist(outdir, ['X.npy', 'Y.npy', 'T.npy', 'P.npy', 'S.npy']):
        print(f"Using files saved at \"{outdir}\".")
        return

    def worker(datafile):
        X, Y, T = make_windows(load_data(datafile), **kwargs)

        pid = Path(datafile)

        for _ in pid.suffixes:
            pid = Path(pid.stem)

        pid = str(pid)

        P = np.array([pid] * len(X))

        return X, Y, T, P

    datafiles = glob(os.path.join(datadir, OXWALK_DATAFILES))

    X, Y, T, P = zip(*Parallel(n_jobs=n_jobs)(
        delayed(worker)(datafile) 
        for datafile in tqdm(datafiles, desc="Load all and make windows")))


    X = np.vstack(X)
    Y = np.hstack(Y)
    T = np.hstack(T)
    P = np.hstack(P)
    S = sources * len(X)

    X_feats = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(extract_features)(x) for x in tqdm(X, desc="Extracting features")))

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'X.npy'), X)
    np.save(os.path.join(outdir, 'Y.npy'), Y)
    np.save(os.path.join(outdir, 'T.npy'), T)
    np.save(os.path.join(outdir, 'P.npy'), P)
    np.save(os.path.join(outdir, 'S.npy'), S)
    X_feats.to_pickle(os.path.join(outdir, 'X_feats.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', default='data')
    parser.add_argument('--outdir', '-o', default='prepared_data')
    parser.add_argument('--sources', '-s', default='oxwalk,ldopa')
    parser.add_argument('--winsec', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=10)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    sources = args.sources.split(',')
    if 'oxwalk' in sources:
        download_oxwalk(args.datadir, args.overwrite)
    
    if 'ldopa' in sources:
        download_ldopa(args.datadir, args.overwrite, args.n_jobs)

    load_all_and_make_windows(args.datadir, args.outdir, args.n_jobs,
                              sources, args.overwrite, winsec=args.winsec)
