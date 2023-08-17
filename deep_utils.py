import numpy as np
import os
import random
import torch
from torch.utils.data.dataset import Dataset
from transforms3d.axangles import axangle2mat
from torchvision import transforms

from utils import get_logger

LOG = get_logger()


class RandomSwitchAxis:
    """
    Randomly switch the three axises for the raw files
    Input size: 3 * FEATURE_SIZE
    """

    def __call__(self, sample):
        # print(sample.shape)
        # 3 * FEATURE
        x = sample[0, :]
        y = sample[1, :]
        z = sample[2, :]

        choice = random.randint(1, 6)

        if choice == 1:
            sample = torch.stack([x, y, z], dim=0)
        elif choice == 2:
            sample = torch.stack([x, z, y], dim=0)
        elif choice == 3:
            sample = torch.stack([y, x, z], dim=0)
        elif choice == 4:
            sample = torch.stack([y, z, x], dim=0)
        elif choice == 5:
            sample = torch.stack([z, x, y], dim=0)
        elif choice == 6:
            sample = torch.stack([z, y, x], dim=0)
        # print(sample.shape)
        return sample


class RotationAxis:
    """
    Rotation along an axis
    """

    def __call__(self, sample):
        # 3 * FEATURE_SIZE
        sample = np.swapaxes(sample, 0, 1)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        axis = np.random.uniform(low=-1, high=1, size=sample.shape[1])
        sample = np.matmul(sample, axangle2mat(axis, angle))
        sample = np.swapaxes(sample, 0, 1)
        return sample


class NormalDataset(Dataset):
    def __init__(
        self,
        X,
        y=None,
        pid=None,
        name="",
        is_labelled=False,
        transform=False,
        transpose_channels_first=True,
        gpu=-1,
    ):
        if transpose_channels_first:
            X = np.transpose(X, (0, 2, 1))

        if gpu != -1:
            my_device = "cuda:" + str(gpu)
        else:
            my_device = "cpu"

        self.X = torch.from_numpy(X).to(my_device, dtype=torch.float)
        if y is not None:
            self.y = torch.tensor(y).to(my_device, dtype=torch.long)
        self.isLabel = is_labelled
        self.pid = pid
        if transform:
            self.transform = transforms.Compose([RandomSwitchAxis(), RotationAxis()])
        else:
            self.transform = None

        LOG.info(name + " set sample count : " + str(len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]

        if self.isLabel:
            y = self.y[idx]
        else:
            y = np.NaN

        if self.pid is not None:
            pid = self.pid[idx]
        else:
            pid = np.NaN

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, y, pid


def get_inverse_class_weights(y):
    """Return a list with inverse class frequencies in y"""
    import collections

    counter = collections.Counter(y)
    for i in range(len(counter)):
        if i not in counter.keys():
            counter[i] = 1

    num_samples = len(y)
    weights = [0] * len(counter)
    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)
    LOG.info("Inverse class weights: ")
    LOG.info(weights)

    return weights


class EarlyStopping:
    """Early stops the training if validation loss
    doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time v
                            alidation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each
                            validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity
                            to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            msg = "Validation loss decreased"
            msg = msg + f" ({self.val_loss_min:.6f} --> {val_loss:.6f}). "
            msg = msg + "Saving model ..."
            self.trace_func(msg)
        if hasattr(model, "module"):
            torch.save(model.module.state_dict(), self.path)
        else:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
