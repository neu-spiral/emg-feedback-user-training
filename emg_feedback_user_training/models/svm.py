import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator


class SVM(BaseEstimator):
    def __init__(self, C=1e3, M=1e0):
        """
        Implementation of Semisupervised Support Vector Machines by Kristin Bennett using cvxpy library:
        Bennett, Kristin, and Ayhan Demiriz. "Semi-supervised support vector machines." from
        Advances in Neural Information processing systems (1999): 368-374.

        Args:
            C (float): Hyperparameter in Equation 7 selected by cross-validation
            M (float): Hyperparameter in Equation 7 selected by cross-validation
        """
        self.C = C
        self.M = M

    def fit(self, X, y):
        """Fit the semisupervised SVM model according to the given training data."""
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.N_labeled = len(y)
        self.N_unlabeled = self.N - self.N_labeled
        self.w = cp.Variable((1, self.d), name="w")
        self.b = cp.Variable(name="b")
        self.eta = cp.Variable((1, self.N_labeled), name="eta")
        if self.N_unlabeled != 0:  # unlabeled data exist
            self.slack = cp.Variable((1, self.N_unlabeled), name="slack")
            self.z = cp.Variable((1, self.N_unlabeled), name="z")
            self.d = cp.Variable((1, self.N_unlabeled), integer=True, name="d")
        else:  # just supervised training
            self.slack = None
            self.z = None
            self.d = None
        objective = cp.Minimize(
            self.C * (cp.sum(self.eta) + (cp.sum(self.slack) + cp.sum(self.z))) + cp.norm(self.w, 2)
        )

        constraints = []
        constraints += [int(y[i]) * (self.w @ X[i] - self.b) + self.eta[0, i] >= 1 for i in range(self.N_labeled)]
        constraints += [self.eta[0, i] >= 0 for i in range(self.N_labeled)]
        if self.N_unlabeled != 0:  # unlabeled data exist:
            constraints += [
                self.w @ X[self.N_labeled + j] - self.b + self.slack[0, j] + self.M * (1 - self.d[0, j]) >= 1
                for j in range(self.N_unlabeled)
            ]
            constraints += [self.slack[0, j] >= 0 for j in range(self.N_unlabeled)]
            constraints += [
                -(self.w @ X[self.N_labeled + j] - self.b) + self.z[0, j] + self.M * self.d[0, j] >= 1
                for j in range(self.N_unlabeled)
            ]
            constraints += [self.z[0, j] >= 0 for j in range(self.N_unlabeled)]
            constraints += [self.d[0, j] <= 1 for j in range(self.N_unlabeled)]
            constraints += [self.d[0, j] >= 0 for j in range(self.N_unlabeled)]

        problem = cp.Problem(objective, constraints)
        problem.solve(warm_start=True)
        return self

    def predict(self, traindata):
        """Calculates the SVM scores for each pairwise SVM for a single input data"""
        score = self.w.value @ traindata.T - self.b.value
        return score

    def __eq__(self, other):
        return np.allclose(self.w.value, other.w.value) and np.allclose(self.b.value, other.b.value)
