from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
import numpy as np
from sklearn.metrics import f1_score

from utils import save_dict
from eval_utils import cross_val_score, f1_macro_score

DEFAULT_SSL_PARAM_GRID = {"learning_rate": hp.uniform("learning_rate", 1e-6, 1e-3)}

DEFAULT_RF_PARAM_GRID = {
    "max_depth": hp.choice("max_depth", np.arange(5, 31, dtype=int)),
    "min_samples_split": hp.choice("min_samples_split", [2, 5, 10, 20]),
    "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 4, 8]),
    "max_features": hp.choice("max_features", ["sqrt", "log2", 0.5, 0.8]),
    "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0.0, 0.3),
    "max_leaf_nodes": hp.choice("max_leaf_nodes", [None, 10, 50, 100]),
}


def optimise_ssl(
    models,
    X,
    y,
    groups=None,
    param_grid=DEFAULT_SSL_PARAM_GRID,
    outdir="optimised_params/ssl.pkl",
):
    def objective(space):
        f1_scores = cross_val_score(
            models,
            X,
            y,
            groups=groups,
            cv=3,
            scoring=f1_macro_score,
        )
        mean_f1 = np.mean(f1_scores)

        return {"loss": -mean_f1, "status": STATUS_OK}

    trials = Trials()

    best = fmin(
        fn=objective,
        space=param_grid,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        verbose=1,
        rstate=np.random.default_rng(42),
    )

    optimised_params = space_eval(param_grid, best)

    save_dict(optimised_params, outdir)

    return optimised_params


def optimise_rf(
    models,
    X,
    y,
    groups=None,
    param_grid=DEFAULT_RF_PARAM_GRID,
    outdir="optimised_params/rf.pkl",
):
    def f1_macro(y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")

    def objective(space):
        f1_scores = cross_val_score(
            models,
            X,
            y,
            groups=groups,
            cv=3,
            scoring=f1_macro,
        )
        mean_f1 = np.mean(f1_scores)
        return {"loss": -mean_f1, "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        fn=objective,
        space=param_grid,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        verbose=1,
        rstate=np.random.default_rng(42),
    )
    optimised_params = space_eval(param_grid, best)

    save_dict(optimised_params, outdir)

    return optimised_params
