import heapq

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from . import recommender

class ItemKNN(recommender.PredictRecommender):
    def __init__(self, shrinkage=0, neighborhood_size=40,
                 user_based=True, use_content=True, use_means=True):
        super().__init__()
        self._shrinkage = shrinkage
        self._neighborhood_size = neighborhood_size
        self._user_based = user_based
        self._use_content = use_content
        self._use_means = use_means
        self._feature_matrix = scipy.sparse.csr_matrix((0, 0))
        self._means = np.empty(0)
        self._similarity_matrix = np.empty((0, 0))

    def reset(self, users=None, items=None, ratings=None):
        self._feature_matrix = scipy.sparse.csr_matrix((0, 0))
        self._similarity_matrix = np.empty((0, 0))
        self._means = np.empty(0)
        super().reset(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):
        super().update(users, items, ratings)
        if self._user_based:
            self._feature_matrix = scipy.sparse.csr_matrix(self._ratings)
        else:
            self._feature_matrix = scipy.sparse.csr_matrix(self._ratings.T)
        self._means = (flatten(self._feature_matrix.sum(axis=1)) /
                       self._feature_matrix.getnnz(axis=1))
        if self._use_content:
            if self._user_based:
                self._feature_matrix = scipy.sparse.hstack([self._feature_matrix, self._users])
            else:
                self._feature_matrix = scipy.sparse.hstack([self._feature_matrix, self._items])
        self._similarity_matrix = cosine_similarity(self._feature_matrix, self._feature_matrix,
                                                    self._shrinkage)

    def _predict(self, user_item):
        preds = []
        for user_id, item_id, _ in user_item:
            if self._user_based:
                relevant_idxs = nlargest_indices(self._neighborhood_size,
                                                  self._similarity_matrix[user_id])
                similarities = self._similarity_matrix[relevant_idxs, user_id]
                ratings = flatten(self._ratings[relevant_idxs, item_id])
                mean = self._means[user_id]
            else:
                relevant_indexes = nlargest_indices(self._neighborhood_size,
                                                  self._similarity_matrix[item_id])
                similarities = self._similarity_matrix[relevant_idxs, item_id]
                ratings = flatten(self._ratings.T[relevant_idxs, user_id])
                mean = self._means[item_id]
            relevant_means = self._means[relevant_idxs]
            nonzero = ratings != 0
            ratings = ratings[nonzero]
            similarities = similarities[nonzero]
            if self._use_means:
                preds.append(mean + np.average(ratings - relevant_means[nonzero],
                                               weights=similarities))
            else:
                preds.append(np.average(ratings, weigths=similarities))

        return np.array(preds)


def cosine_similarity(X, Y, shrinkage):
    return (X @ Y.T).A / (scipy.sparse.linalg.norm(X, axis=1)[:, np.newaxis] *
                          scipy.sparse.linalg.norm(Y, axis=1)[np.newaxis, :] + shrinkage)


def nlargest_indices(n, iterable):
    nlargest = heapq.nlargest(n, enumerate(iterable),
                              key=lambda x: x[1])
    return [i[0] for i in nlargest]


def flatten(matrix):
    return matrix.A.flatten()
