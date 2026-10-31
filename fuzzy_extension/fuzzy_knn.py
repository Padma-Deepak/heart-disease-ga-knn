"""Fuzzy KNN with neighbour-initialized training memberships.

Keller, Gray and Givens (1985), doi:10.1109/TSMC.1985.6313426.
Our explicit choices: exclude self from initialization, fixed initialization k=5,
stable training-row order for equal distances, inverse-distance exponent 2/(m-1).
Membership scores are not calibrated medical probabilities.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

def training_memberships(X, y, metric, init_k=5):
    distances = pairwise_distances(X, metric=metric)
    np.fill_diagonal(distances, np.inf)
    neighbours = np.argsort(distances, axis=1, kind='stable')[:, :init_k]
    frequencies = np.eye(2)[y[neighbours]].mean(axis=1)
    memberships = .51 * np.eye(2)[y] + .49 * frequencies
    return memberships

def nearest_geometry(Xtrain, Xquery, metric, max_k):
    distances = pairwise_distances(Xquery, Xtrain, metric=metric)
    indices = np.argsort(distances, axis=1, kind='stable')[:, :max_k]
    return np.take_along_axis(distances, indices, axis=1), indices

def combine_memberships(distances, indices, memberships, k, fuzziness):
    distances, indices = distances[:, :k], indices[:, :k]
    zeros = distances <= 1e-12
    # Log-space normalization avoids overflow for m close to one.
    logs = np.log(np.maximum(distances, 1e-12))
    logs -= logs.min(axis=1, keepdims=True)
    weights = np.exp(-2. / (fuzziness-1.) * logs)
    has_zero = zeros.any(axis=1)
    weights[has_zero] = zeros[has_zero]
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum('qk,qkc->qc', weights, memberships[indices])

class FuzzyKNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors=5, metric='euclidean', fuzziness=2., init_k=5):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.fuzziness = fuzziness
        self.init_k = init_k

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError('This implementation expects both binary classes 0 and 1.')
        if self.fuzziness <= 1 or not 1 <= self.n_neighbors <= len(X) or not 1 <= self.init_k < len(X):
            raise ValueError('Invalid k, initialization k or fuzziness.')
        self.X_ = X.copy()
        self.y_ = np.asarray(y, dtype=int).copy()
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.memberships_ = training_memberships(self.X_, self.y_, self.metric, self.init_k)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'memberships_')
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Feature count differs from fitted data.')
        distances, indices = nearest_geometry(self.X_, X, self.metric, self.n_neighbors)
        return combine_memberships(distances, indices, self.memberships_, self.n_neighbors, self.fuzziness)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
