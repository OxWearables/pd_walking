import argparse
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np
import os
import pandas as pd
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
from utils import load_dict, save_dict


DEFAULT_ATTRIBUTES = {
    "n_estimators": 3000,
    "replacement": True,
    "sampling_strategy": "not minority",
    "oob_score": True,
    "n_jobs": 12,
    "random_state": 42,
    "max_features": "sqrt",
}

DEFAULT_PARAM_GRID = {
    "max_depth": hp.choice("max_depth", np.arange(5, 31, dtype=int)),
    "min_samples_split": hp.choice("min_samples_split", [2, 5, 10, 20]),
    "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 4, 8]),
    "max_features": hp.choice("max_features", ["sqrt", "log2", 0.5, 0.8]),
    "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0.0, 0.3),
    "max_leaf_nodes": hp.choice("max_leaf_nodes", [None, 10, 50, 100]),
}


def load_rf(optimisedir="", **kwargs):
    optimised_attributes = (
        load_dict(optimisedir) if optimisedir and os.path.exists(optimisedir) else {}
    )

    model_attributes = {**DEFAULT_ATTRIBUTES, **optimised_attributes, **kwargs}

    model = BalancedRandomForestClassifier(**model_attributes)

    return model


class RandomForestClassifierWrapper:
    def __init__(self, **kwargs):
        self.model = load_rf(**kwargs)

    def __str__(self):
        return (
            "Random Forest Classifier:\n"
            f"  Model Parameters: {self.model.get_params()}\n"
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def optimise(
        self,
        X,
        y,
        groups=None,
        param_grid=DEFAULT_PARAM_GRID,
        outdir="optimised_params/rf.pkl",
    ):
        def objective(space):
            clf = load_rf(**space, oob_score=False)

            f1_scores = cross_val_score(
                clf,
                X,
                y,
                groups=groups,
                cv=3,
                scoring=make_scorer(f1_score, average="macro"),
            )
            mean_f1 = np.mean(f1_scores)

            return {"loss": -mean_f1, "status": STATUS_OK}

        trials = Trials()

        best = fmin(
            fn=objective,
            space=param_grid,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            verbose=1,
            rstate=np.random.default_rng(42),
        )

        optimised_params = space_eval(param_grid, best)

        save_dict(optimised_params, outdir)

        return optimised_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", "-d", default="prepared_data")
    parser.add_argument("--optimisedir", "-o", default="optimised_params")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    X = pd.read_pickle(os.path.join(args.datadir, "X_feats.pkl")).values
    y = np.load(os.path.join(args.datadir, "Y.npy"))
    P = np.load(os.path.join(args.datadir, "P.npy"))

    if args.smoke_test:
        np.random.seed(42)
        smoke_idx = np.random.randint(len(y), size=int(0.01 * len(y)))

        X, y, P = X[smoke_idx], y[smoke_idx], P[smoke_idx]

    rf = RandomForestClassifierWrapper()
    smoke_flag = "_smoke" if args.smoke_test else ""
    params = rf.optimise(
        X, y, P, outdir=f"{args.optimisedir}/rf_{args.annot}{smoke_flag}.pkl"
    )

    print(f"Best params: {params}")
