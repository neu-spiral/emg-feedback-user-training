import numpy as np
import sklearn
from sklearn.multiclass import OneVsOneClassifier as _OVO
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _num_samples, check_is_fitted


def _fit_ovo_binary_patched(estimator, X, y, i, j):
    """Slightly modified version of sklearn.multiclass._fit_ovo_binary.
    Fit a single binary estimator (one-vs-one)."""
    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    y_binary = np.empty(y.shape, int)
    y_binary[y == i] = -1  # -1 instead of 0
    y_binary[y == j] = 1
    indcond = np.arange(_num_samples(X))[cond]
    return (
        sklearn.multiclass._fit_binary(
            estimator,
            _safe_split(estimator, X, None, indices=indcond)[0],
            y_binary,
            classes=[i, j],
        ),
        indcond,
    )


# Monkey patch before class definition
sklearn.multiclass._fit_ovo_binary = _fit_ovo_binary_patched


class OneVsOneClassifier(_OVO):
    """Slightly modified version of sklearn.multiclass.OneVsOneClassifier."""

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )

        indices = self.pairwise_indices_
        if indices is None:
            Xs = [X] * len(self.estimators_)
        else:
            Xs = [X[:, idx] for idx in indices]

        predictions = np.vstack([est.predict(Xi) for est, Xi in zip(self.estimators_, Xs)]).T
        return predictions
