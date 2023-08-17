import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import os
import logging
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from torch.utils.data import DataLoader
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval


from utils import dotdict, load_environment_vars, load_dict, save_dict
from eval_utils import cross_val_score, f1_macro_score
from deep_utils import (
    get_logger,
    get_inverse_class_weights,
    NormalDataset,
    EarlyStopping,
)

LOG = get_logger()

DEFAULT_SSL_CONFIG = {
    "ssl": {
        "augmentation": True,
        "weighted_loss_fn": True,
        "learning_rate": 0.0001,
        "patience": 10,
        "num_epoch": 100,
    }
}

DEFAULT_PARAM_GRID = {
    "learning_rate": hp.uniform("learning_rate", 1e-6, 1e-3),
    "batch_size": hp.choice("batch_size", [100, 1000, 5000, 10000]),
    "pretrained": hp.choice("pretrained", [True, False]),
}

SSL_REPO_PATH, GPU = load_environment_vars(["SSL_REPO_PATH", "GPU"])


def get_sslnet(device, cfg, ssl_weights_path=None, pretrained=False):
    """
    Load and return the SSLNet.

    :param str device: pytorch map location
    :param cfg: config object
    :param ssl_weights_path: the path of the pretrained (fine-tuned) weights.
    :param bool pretrained: Initialise the model with self-supervised pretrained weights.
    :return: pytorch SSLNet model
    :rtype: nn.Module
    """
    if cfg.ssl_repo_path:
        # use repo from disk (for offline use)
        LOG.info("Using local %s", cfg.ssl_repo_path)
        sslnet: nn.Module = torch.hub.load(
            cfg.ssl_repo_path,
            f"harnet{cfg.data.winsec}",
            source="local",
            class_num=cfg.data.output_size,
            pretrained=pretrained,
        )
    else:
        # download repo from github
        repo = "OxWearables/ssl-wearables"
        sslnet: nn.Module = torch.hub.load(
            repo,
            f"harnet{cfg.data.winsec}",
            trust_repo=True,
            class_num=cfg.data.output_size,
            pretrained=pretrained,
        )

    if ssl_weights_path:
        # load pretrained weights
        model_dict = torch.load(ssl_weights_path, map_location=device)
        sslnet.load_state_dict(model_dict)
        LOG.info("Loaded SSLNet weights from %s", ssl_weights_path)

    sslnet.to(device)

    return sslnet


def predict(model, data_loader, my_device, output_logits=False):
    """
    Iterate over the dataloader and do inference with a pytorch model.

    :param nn.Module model: pytorch Module
    :param data_loader: pytorch dataloader
    :param str my_device: pytorch map device
    :param bool output_logits: When True, output the raw outputs (logits) from the last layer (before classification).
                                When False, argmax the logits and output a classification scalar.
    :return: true labels, model predictions, pids
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """

    predictions_list = []
    true_list = []
    pid_list = []
    model.eval()
    if my_device == "cpu":
        torch.set_flush_denormal(True)
    for i, (x, y, pid) in enumerate(
        tqdm(data_loader, mininterval=60, disable=LOG.getEffectiveLevel() > 20)
    ):
        with torch.inference_mode():
            x = x.to(my_device, dtype=torch.float)
            logits = model(x)
            true_list.append(y)
            if output_logits:
                predictions_list.append(logits.cpu())
            else:
                pred_y = torch.argmax(logits, dim=1)
                predictions_list.append(pred_y.cpu())
            pid_list.extend(pid)
    true_list = torch.cat(true_list)
    predictions_list = torch.cat(predictions_list)

    if output_logits:
        return (
            torch.flatten(true_list).numpy(),
            predictions_list.numpy(),
            np.array(pid_list),
        )
    else:
        return (
            torch.flatten(true_list).numpy(),
            torch.flatten(predictions_list).numpy(),
            np.array(pid_list),
        )


def train(model, train_loader, val_loader, cfg, my_device, weights, weights_path):
    """
    Iterate over the training dataloader and train a pytorch model.
    After each epoch, validate model and early stop when validation loss function bottoms out.

    Trained model weights will be saved to disk (cfg.ssl.weights).

    :param nn.Module model: pytorch model
    :param train_loader: training data loader
    :param val_loader: validation data loader
    :param cfg: config object.
    :param str my_device: pytorch map device.
    :param weights: training class weights
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.ssl.learning_rate, amsgrad=True
    )

    if cfg.ssl.weighted_loss_fn:
        weights = torch.FloatTensor(weights).to(my_device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(
        patience=cfg.ssl.patience, path=weights_path, verbose=True, trace_func=LOG.info
    )

    for epoch in range(cfg.ssl.num_epoch):
        model.train()
        train_losses = []
        train_acces = []
        for i, (x, y, _) in enumerate(
            tqdm(train_loader, disable=LOG.getEffectiveLevel() > 20)
        ):
            x.requires_grad_(True)
            x = x.to(my_device, dtype=torch.float)
            true_y = y.to(my_device, dtype=torch.long)

            optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, true_y)
            loss.backward()
            optimizer.step()

            pred_y = torch.argmax(logits, dim=1)
            train_acc = torch.sum(pred_y == true_y)
            train_acc = train_acc / (pred_y.size()[0])

            train_losses.append(loss.cpu().detach())
            train_acces.append(train_acc.cpu().detach())

        val_loss, val_acc = _validate_model(model, val_loader, my_device, loss_fn)

        epoch_len = len(str(cfg.ssl.num_epoch))
        print_msg = (
            f"[{epoch:>{epoch_len}}/{cfg.ssl.num_epoch:>{epoch_len}}] | "
            + f"train_loss: {np.mean(train_losses):.3f} | "
            + f"train_acc: {np.mean(train_acces):.3f} | "
            + f"val_loss: {val_loss:.3f} | "
            + f"val_acc: {val_acc:.2f}"
        )

        early_stopping(val_loss, model)
        LOG.info(print_msg)

        if early_stopping.early_stop:
            LOG.info("Early stopping")
            LOG.info("SSLNet weights saved to %s", weights_path)
            break

    return model


def _validate_model(model, val_loader, my_device, loss_fn):
    """Iterate over a validation data loader and return mean model loss and accuracy."""
    model.eval()
    losses = []
    acces = []
    for i, (x, y, _) in enumerate(val_loader):
        with torch.inference_mode():
            x = x.to(my_device, dtype=torch.float)
            true_y = y.to(my_device, dtype=torch.long)

            logits = model(x)
            loss = loss_fn(logits, true_y)

            pred_y = torch.argmax(logits, dim=1)

            val_acc = torch.sum(pred_y == true_y)
            val_acc = val_acc / (list(pred_y.size())[0])

            losses.append(loss.cpu().detach())
            acces.append(val_acc.cpu().detach())
    losses = np.array(losses)
    acces = np.array(acces)
    return np.mean(losses), np.mean(acces)


class SSLClassifier:
    def __init__(
        self,
        class_labels,
        weights_path,
        winsec=10,
        batch_size=10000,
        pretrained=True,
        load_weights=True,
        verbose=True,
        seed=42,
        learning_rate=0,
        optimisedir="",
        fold=0,
    ):
        """
        SSLClassifier is a class for training and using SSLNet-based classifiers.

        Args:
            class_labels (list): List of class labels.
            weights_path (str): Path to the SSLNet weights file.
            winsec (float): Window size in seconds.
            batch_size (int): Batch size for training and inference.
            pretrained (bool): Whether to use pretrained weights.
            load_weights (bool): When to load weights from saved weights_path.
            verbose (bool): Whether to enable verbose logging.
            seed (int): Random seed for reproducibility.
            learning_rate (float): Learning rate during training.
            optimisedir (str): Path to the optimised parameters file.
            fold (int): Integer representing the cross validation fold number.
        """
        le = LabelEncoder()
        le.fit(class_labels)

        self.le = le
        self.class_labels = class_labels

        optimised_lr = (
            load_dict(optimisedir)
            if optimisedir and os.path.exists(optimisedir)
            else {}
        )

        cfg = {
            "ssl_repo_path": SSL_REPO_PATH,
            "data": {
                "output_size": len(class_labels),
                "winsec": winsec,
            },
            **DEFAULT_SSL_CONFIG,
        }

        if learning_rate != 0:
            cfg["ssl"]["learning_rate"] = learning_rate
        elif optimised_lr and "learning_rate" in optimised_lr:
            cfg["ssl"]["learning_rate"] = optimised_lr["learning_rate"]

        self.verbose = verbose

        if verbose:
            LOG.setLevel(logging.DEBUG)
        else:
            LOG.setLevel(logging.CRITICAL)

        self.cfg = dotdict(cfg)

        self.my_device = f"cuda:{GPU}" if GPU != -1 else "cpu"

        self.weights_path = weights_path.format(fold)
        self.pretrained = pretrained
        self.seed = seed
        self.batch_size = batch_size

        if load_weights & os.path.exists(self.weights_path):
            self.model = get_sslnet(
                self.my_device, self.cfg, self.weights_path, pretrained
            )

        else:
            self.model = get_sslnet(self.my_device, self.cfg, None, pretrained)

    def __str__(self):
        return (
            f"SSLClassifier:\n"
            f"  Class Labels: {self.le.classes_}\n"
            f"  SSL Repo Path: {self.cfg.ssl_repo_path}\n"
            f"  Weights Path: {self.weights_path}\n"
            f"  Pretrained: {self.pretrained}\n"
            f"  Winsec: {self.cfg.data.winsec}\n"
            f"  Device: {self.my_device}\n"
        )

    def predict_proba(self, data):
        """
        Compute class probabilities for the given data.

        Args:
            data (str or pd.DataFrame): Input data. Either a file path or a DataFrame.
            batch_size (int): Batch size for inference.

        Returns:
            np.ndarray: Array of class probabilities with shape (n_samples, n_classes).
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        dataset = NormalDataset(data)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        LOG.info("SSL inference")
        _, y_logits, _ = predict(
            self.model, dataloader, self.my_device, output_logits=True
        )

        y_score = softmax(y_logits, axis=1)

        return y_score

    def predict_from_proba(self, y_score):
        """
        Predict class labels from class probabilities.

        Args:
            y_score (np.ndarray): Array of class probabilities with shape (n_samples, n_classes).

        Returns:
            np.ndarray: Array of predicted class labels.
        """
        y_pred = np.argmax(y_score, axis=1)

        return self.le.inverse_transform(y_pred)

    def predict(self, data):
        """
        Predict class labels for the given data.

        Args:
            data (str or pd.DataFrame): Input data. Either a file path or a DataFrame.

        Returns:
            np.ndarray: Array of predicted class labels.
        """
        y_score = self.predict_proba(data)

        return self.predict_from_proba(y_score)

    def fit(self, X, y, groups=None, val_split=0.125):
        """
        Fit the SSLNet model to the labeled data.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target class labels.
            pids (np.ndarray or None): Patient IDs corresponding to the data samples.
            weights_path (str or None): Path to save the trained weights.
            hmm_weights_path (str or None): Path to save the HMM weights.
            val_split (float): Fraction of data to use for validation.
            overwrite (bool): Whether to overwrite existing weight files.
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if groups is not None:
            unique_pids = np.unique(groups)
            val_ids = np.random.choice(
                unique_pids, int(val_split * len(unique_pids)), replace=False
            )
            mask_val = np.isin(groups, val_ids)

        else:
            num_samples = len(X)
            val_ids = np.random.choice(
                num_samples, int(val_split * num_samples), replace=False
            )
            mask_val = np.zeros(num_samples, dtype=bool)
            mask_val[val_ids] = True

        mask_train = ~mask_val

        y = self.le.transform(y)

        X_train, y_train = X[mask_train], y[mask_train]
        X_val, y_val = X[mask_val], y[mask_val]

        # construct train and validation dataloaders
        train_dataset = NormalDataset(
            X_train,
            y_train,
            name="train",
            is_labelled=True,
            transform=self.cfg.ssl.augmentation,
        )
        val_dataset = NormalDataset(X_val, y_val, name="val", is_labelled=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        sslnet = self.model

        if not self.weights_path:
            raise Exception("No weights path provided")

        LOG.info("SSLNet training")
        train(
            sslnet,
            train_loader,
            val_loader,
            self.cfg,
            self.my_device,
            get_inverse_class_weights(y_train),
            self.weights_path,
        )
        LOG.info("SSLNet weights saved to %s", self.weights_path)

        # update model to saved best weights from training
        self.model = get_sslnet(self.my_device, self.cfg, self.weights_path, False)

    def optimise(
        self,
        X,
        y,
        groups=None,
        param_grid=DEFAULT_PARAM_GRID,
        weightsdir="model_weights/ssl_opt_{}.pt",
        outdir="optimised_params/ssl.pkl",
    ):
        def objective(space):
            models = [
                SSLClassifier(
                    self.class_labels, weightsdir, fold=i, load_weights=False, **space
                )
                for i in range(3)
            ]

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
            max_evals=20,
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
    parser.add_argument("--optimisedir", "-o", default="outputs/optimised_params")
    parser.add_argument("--weightsdir", "-w", default="outputs/model_weights")
    parser.add_argument("--smoke_test", default=True)  # action="store_true")
    args = parser.parse_args()

    X = np.load(os.path.join(args.datadir, "X.npy"))
    y = np.load(os.path.join(args.datadir, "Y.npy"))
    P = np.load(os.path.join(args.datadir, "P.npy"))

    if args.smoke_test:
        np.random.seed(42)
        smoke_idx = np.random.randint(len(y), size=int(0.01 * len(y)))

        X, y, P = X[smoke_idx], y[smoke_idx], P[smoke_idx]

    smoke_flag = "_smoke" if args.smoke_test else ""
    ssl = SSLClassifier(
        np.unique(y), os.path.join(args.weightsdir, f"ssl_opt{smoke_flag}_{{}}.pt")
    )
    params = ssl.optimise(
        X,
        y,
        P,
        weightsdir=f"{args.weightsdir}/ssl_opt_{{}}.pt",
        outdir=f"{args.optimisedir}/ssl{smoke_flag}.pkl",
    )

    print(f"Best params: {params}")
