import numpy as np
import pandas as pd
import argparse
import os

from eval_utils import cross_val_predict, fit_predict
from classifier import Classifier


def evaluate_model(
    model_type,
    X,
    y,
    groups,
    train_mask=None,
    test_mask=None,
    cv=0,
    n_jobs=None,
    outFilePath="",
    **kwargs,
):
    if train_mask is not None:
        X_train, y_train, groups_train = (
            X[train_mask],
            y[train_mask],
            groups[train_mask],
        )

    else:
        X_train, y_train, groups_train = X, y, groups

    if cv == 0:
        if test_mask is not None:
            X_test = X[test_mask]

        else:
            X_test = X

        model = Classifier(model_type, **kwargs)
        y_pred = fit_predict(model, X_train, y_train, X_test, **kwargs)

    else:
        n_splits = cv if isinstance(cv, int) else cv.get_n_splits()

        if n_splits > 0:
            models = [
                Classifier(model_type, fold, **kwargs) for fold in range(n_splits)
            ]

            y_pred = cross_val_predict(
                models, X_train, y_train, groups_train, cv=cv, n_jobs=n_jobs
            )

            if test_mask is not None:
                if train_mask is not None:
                    y_pred = y_pred[test_mask[train_mask]]

                else:
                    y_pred = y_pred[test_mask]

        else:
            raise ValueError("cv must be natural number")

    if outFilePath:
        os.makedirs(os.path.dirname(outFilePath), exist_ok=True)
        np.save(outFilePath, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", "-d", default="prepared_data")
    parser.add_argument("--train_source", default="")
    parser.add_argument("--test_source", default="")
    parser.add_argument("--cv", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument("--model_types", "-m", default="rf")
    args = parser.parse_args()

    X = pd.read_pickle(os.path.join(args.datadir, "X_feats.pkl")).values
    y = np.load(os.path.join(args.datadir, "Y.npy"))
    P = np.load(os.path.join(args.datadir, "P.npy"))
    S = np.load(os.path.join(args.datadir, "S.npy"))

    train_source = args.train_source.upper().split(",")
    test_source = args.test_source.upper().split(",")
    train_mask = np.isin(S, train_source) if args.train_source else None
    test_mask = np.isin(S, test_source) if args.test_source else None

    train_source = "".join(train_source) or "all"
    test_source = "".join(test_source) or "all"

    for model_type in args.model_types.split(","):
        fileName = f"y_pred_{model_type}_train_{train_source}_test_{test_source}.npy"

        evaluate_model(
            model_type,
            X,
            y,
            P,
            train_mask,
            test_mask,
            args.cv,
            args.n_jobs,
            os.path.join("outputs", "predictions", fileName),
        )
