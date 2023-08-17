import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import indexable
from sklearn.metrics import accuracy_score, f1_score

from utils import flatten_list


def fit_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)


def cross_val_predict(models, X, y, groups=None, cv=5, n_jobs=None):
    X, y, groups = indexable(X, y, groups)

    if isinstance(cv, int):
        cv = StratifiedGroupKFold(n_splits=cv)

    if len(models) != cv.get_n_splits():
        raise ValueError("cross validation requires a model for each fold")

    splits = list(cv.split(X, y, groups))

    test_indices = np.concatenate([test for _, test in splits])

    predictions = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_predict)(models[i], X, y, train, test)
        for i, (train, test) in enumerate(splits)
    )

    predictions_with_indices = list(zip(flatten_list(predictions), test_indices))
    predictions_with_indices.sort(key=lambda x: x[1])

    sorted_predictions = np.array([pred for pred, _ in predictions_with_indices])

    return sorted_predictions


def _fit_and_predict(model, X, y, train_idx, test_idx):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test = X[test_idx]

    return fit_predict(model, X_train, y_train, X_test)


def cross_val_score(models, X, y, groups=None, cv=5, n_jobs=None, scoring=None):
    X, y, groups = indexable(X, y, groups)

    if isinstance(cv, int):
        cv = StratifiedGroupKFold(n_splits=cv)

    if len(models) != cv.get_n_splits():
        raise ValueError("cross validation requires a model for each fold")

    splits = list(cv.split(X, y, groups))

    test_indices = np.concatenate([test for _, test in splits])

    scores = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_score)(models[i], X, y, train, test, scoring)
        for i, (train, test) in enumerate(splits)
    )

    scores_with_indices = list(zip(scores, test_indices))
    scores_with_indices.sort(key=lambda x: x[1])

    sorted_scores = np.array([score for score, _ in scores_with_indices])

    return sorted_scores


def _fit_and_score(model, X, y, train_idx, test_idx, scoring):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    y_pred = fit_predict(model, X_train, y_train, X_test)

    if scoring is None:
        scoring = accuracy_score

    return scoring(y_test, y_pred)


def f1_macro_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")
