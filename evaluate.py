import numpy as np
import pandas as pd
import argparse
import os
from glob import glob
from sklearn.metrics import f1_score

from eval_utils import cross_val_predict, fit_predict
from classifier import Classifier


def evaluate_model(
    model_type,
    X,
    y,
    groups,
    train_source=[],
    test_source=[],
    cv=0,
    n_jobs=None,
    outdir="",
    **kwargs,
):
    train_mask = np.isin(S, train_source) if any(train_source) else []
    test_mask = np.isin(S, test_source) if any(test_source) else []

    if any(train_mask):
        X_train, y_train, groups_train = (
            X[train_mask],
            y[train_mask],
            groups[train_mask],
        )

    else:
        X_train, y_train, groups_train = X, y, groups

    if cv == 0:
        X_test = X[test_mask] if any(test_mask) else X
        y_test = y[test_mask] if any(test_mask) else y
        groups_test = groups[test_mask] if any(test_mask) else groups

        model = Classifier(model_type, **kwargs)
        y_pred = fit_predict(model, X_train, y_train, X_test)

    else:
        n_splits = cv if isinstance(cv, int) else cv.get_n_splits()

        if n_splits > 0:
            models = [
                Classifier(model_type, fold, **kwargs) for fold in range(n_splits)
            ]

            y_pred = cross_val_predict(
                models, X_train, y_train, groups_train, cv=cv, n_jobs=n_jobs
            )

            if any(test_mask):
                if any(train_mask):
                    y_pred = y_pred[test_mask[train_mask]]
                    y_test = y[test_mask & train_mask]
                    groups_test = groups[test_mask & train_mask]

                else:
                    y_pred = y_pred[test_mask]
                    y_test = y[test_mask]
                    groups_test = groups[test_mask]

            else:
                y_test = y
                groups_test = groups

        else:
            raise ValueError("number of splits must be natural number")

    scores = pd.Series(
        [
            f1_score(y_test[groups_test == group], y_pred[groups_test == group])
            for group in np.unique(groups_test)
        ],
        index=np.unique(groups_test),
    )

    if outdir:
        train_source = "".join(train_source) or "all"
        test_source = "".join(test_source) or "all"

        os.makedirs(outdir, exist_ok=True)
        np.save(
            os.path.join(
                outdir,
                f"y_pred_{model_type}_train_{train_source}_test_{test_source}.npy",
            ),
            y_pred,
        )
        scores.to_pickle(
            os.path.join(
                outdir,
                f"scores_{model_type}_train_{train_source}_test_{test_source}.pkl",
            )
        )


def join_scores(predictdir):
    df = pd.concat(
        {
            os.path.basename(file).split(".")[0]: pd.Series(pd.read_pickle(file))
            for file in glob(os.path.join(predictdir, "*.pkl"))
            if "all_scores.pkl" not in file
        },
        axis=1,
    )

    df.to_pickle(os.path.join(predictdir, "all_scores.pkl"))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", "-d", default="prepared_data")
    parser.add_argument("--train_source", default="")
    parser.add_argument("--test_source", default="")
    parser.add_argument("--cv", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument("--model_types", "-m", default="rf,ssl")
    args = parser.parse_args()

    X_raw = np.load(os.path.join(args.datadir, "X.npy"))
    X_feats = pd.read_pickle(os.path.join(args.datadir, "X_feats.pkl")).values
    y = np.load(os.path.join(args.datadir, "Y.npy"))
    P = np.load(os.path.join(args.datadir, "P.npy"))
    S = np.load(os.path.join(args.datadir, "S.npy"))

    for model_type in args.model_types.split(","):
        if model_type.upper() == "RF":
            X = X_feats
            n_jobs = args.n_jobs
            kwargs = {
                "optimisedir": os.path.join("outputs", "optimised_params", "rf.pkl")
            }

        elif model_type.upper() == "SSL":
            X = X_raw
            n_jobs = 1
            train_label = "".join(args.train_source) or "all"
            test_label = "".join(args.test_source) or "all"
            kwargs = {
                "class_labels": np.unique(y),
                "weights_path": os.path.join(
                    "outputs",
                    "model_weights",
                    f"ssl_{train_label}_{test_label}_{{}}.pt",
                ),
                "optimisedir": os.path.join("outputs", "optimised_params", "ssl.pkl"),
                "load_weights": False,
            }

        evaluate_model(
            model_type,
            X,
            y,
            P,
            args.train_source.upper().split(","),
            args.test_source.upper().split(","),
            args.cv,
            n_jobs,
            os.path.join("outputs", "predictions"),
            **kwargs,
        )
