import numpy as np
import pandas as pd
import argparse
import os
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import indexable

from classifier import Classifier
from utils import flatten_list


def evaluate_model(model_type, X, y, groups, train_mask=None, test_mask=None, 
                   cv=0, n_jobs=None, outFilePath='', **kwargs):
    if train_mask is not None:
        X, y, P = X[train_mask], y[train_mask], P[train_mask]

    if cv == 0:
        if test_mask is not None:
            X_test = X[test_mask]
        
        else:
            X_test = X
        
        model = Classifier(model_type, **kwargs)
        model.fit(X, y)
        y_pred = model.predict(X_test)

    elif cv > 0:
        y_pred = cross_val_predict(model_type, X, y, groups=groups, cv=cv, n_jobs=n_jobs, **kwargs)

        if test_mask is not None:
            y_pred = y_pred[test_mask]
    
    else:
        raise ValueError("cv must be natural number")

    if outFilePath:
        os.makedirs(os.path.dirname(outFilePath), exist_ok=True)
        np.save(outFilePath, y_pred)


def cross_val_predict(model_type, X, y=None, groups=None, cv=5, n_jobs=None, **kwargs):
    X, y, groups = indexable(X, y, groups)

    cv = StratifiedGroupKFold(n_splits=cv)

    splits = list(cv.split(X, y, groups))

    test_indices = np.concatenate([test for _, test in splits])

    predictions = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_predict)(
            model_type, X, y, train, test, **kwargs
        )
        for train, test in splits
    )

    predictions_with_indices = list(zip(flatten_list(predictions), test_indices))
    predictions_with_indices.sort(key=lambda x: x[1])

    sorted_predictions = np.array([pred for pred, _ in predictions_with_indices])

    return sorted_predictions


def _fit_and_predict(model_type, X, y, train_idx, test_idx, **kwargs):
    model = Classifier(model_type, **kwargs)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test = X[test_idx]

    model.fit(X_train, y_train)

    return model.predict(X_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', default='prepared_data')
    parser.add_argument('--train_source', default='')
    parser.add_argument('--test_source', default='')
    parser.add_argument('--cv', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=10)
    parser.add_argument('--model_types', '-m', default='rf')
    args = parser.parse_args()

    X = pd.read_pickle(os.path.join(args.datadir, "X_feats.pkl")).values
    y = np.load(os.path.join(args.datadir, "Y.npy"))
    P = np.load(os.path.join(args.datadir, "P.npy"))
    S = np.load(os.path.join(args.datadir, "S.npy"))

    train_source = args.train_source.upper().split(',')
    test_source = args.test_source.upper().split(',')
    train_mask = np.isin(S, train_source) if args.train_source else None
    test_mask = np.isin(S, test_source) if args.test_source else None

    train_source = ''.join(train_source) or 'all'
    test_source = ''.join(test_source) or 'all'

    for model_type in args.model_types.split(','):
        fileName = f"y_pred_{model_type}_train_{train_source}_test_{test_source}.npy"

        evaluate_model(model_type, X, y, P, train_mask, test_mask, args.cv, args.n_jobs,
                       os.path.join("outputs", "predictions", fileName))
