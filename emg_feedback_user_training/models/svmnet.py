import pickle
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from emg_feedback_user_training.utils import DATA_DTYPE, NUM_CLASSES

from .ovo import OneVsOneClassifier
from .svm import SVM

DEFAULT_DEVICE = torch.device("cpu")
warnings.simplefilter("ignore")


class RenameUnpickler(pickle.Unpickler):
    # NOTE - accounts for changes in module structure between saving and loading
    def find_class(self, module, name):
        try:
            # For unchanged modules (e.g. numpy or stdlib modules) - use default way
            return super().find_class(module, name)
        except ModuleNotFoundError:
            # For custom classes that have moved - provide new path
            if name == "OneVsOneClassifier":
                return super().find_class("emg_feedback_user_training.models.ovo", name)
            elif name == "SVM":
                return super().find_class("emg_feedback_user_training.models.svm", name)
            else:
                raise NotImplementedError()


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


class SVMNet(nn.Module):
    def __init__(
        self,
        n_classes: int = NUM_CLASSES,
        C=1e3,
        M=1e0,
        batch_size: int = 20,
        epochs: int = 2000,
        device=DEFAULT_DEVICE,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.C = C
        self.M = M
        self.batch_size = batch_size
        self.epochs = epochs
        self.ovo_svm = None
        self.classifier = None
        self.device = device

    def get_weights(self):
        return {"ovo_svm": deepcopy(self.ovo_svm), "classifier": self.classifier.state_dict()}

    def set_weights(self, params):
        self.ovo_svm = params["ovo_svm"]
        # FIXME - hacky solution
        in_dim = params["classifier"]["0.weight"].shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, self.n_classes),
            nn.LogSoftmax(dim=1),
        )
        self.classifier.load_state_dict(params["classifier"])
        return self

    def save_weights(self, filepath: Union[Path, str]):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.get_weights(), f)
        return self

    def load_weights(self, filepath: Union[Path, str]):
        with open(Path(filepath), "rb") as f:
            params = renamed_load(f)
        self.set_weights(params)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Two-stage fitting procedure.
        - fit SVM feature extractor
        - fit linear layer classifier

        X - numpy array of shape (n_samples, n_features)
        y - numpy array of shape (n_samples, )
        """
        assert X.shape[0] == y.shape[0], "Shape mismatch"
        assert y.ndim == 1, "Provide integer labels"

        self.ovo_svm = OneVsOneClassifier(SVM(C=self.C, M=self.M), n_jobs=None)
        # NOTE - fit OVO SVMs first, so we know how many features will be sent to linear layer
        self.ovo_svm.fit(X, y)
        in_dim = (self.ovo_svm.n_classes_ * (self.ovo_svm.n_classes_ - 1)) // 2
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, self.n_classes),
            nn.LogSoftmax(dim=1),
        ).to(self.device)

        optimizer = AdamW(self.classifier.parameters(), lr=1e-4)
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True, num_workers=0)

        self.classifier.train()
        for _ in range(self.epochs):
            for data, labels in dataloader:
                labels = labels.to(self.device)
                log_probs = self.forward(data)

                loss = F.nll_loss(log_probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(-1)

    @torch.no_grad()
    def predict_proba(self, X):
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        self.classifier.eval()
        probs = self.forward(X).cpu().exp().numpy().astype(DATA_DTYPE)
        return np.squeeze(probs)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = self.ovo_svm.predict(x)
        x = torch.from_numpy(x).float().to(self.device)
        x = self.classifier(x)
        return x
