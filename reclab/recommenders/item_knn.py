import heapq

import scipy.sparse
from sklearn.neighbors import NearestNeighbors

from . import recommender

def ItemKNN(recommender.PredictRecommender):
    def __init__(self, shrinkage=0, neighborhood_size=40, user_based=True, use_content=True):
        super().__init__()
        self._shrinkage = shrinkage
        self._neighborhood_size = neighborhood_size
        self._user_based = user_based
        self._use_content = use_content
        self._feature_matrix = scipy.sparse.csr_matrix((0, 0))
        self._means = np.empty(0)
        self._similarity_matrix = np.empty((0, 0))

    def reset(self):
        self._feature_matrix = scipy.sparse.csr_matrix((0, 0))
        self._similarity_matrix = np.empty((0, 0))
        self._means = np.empty(0)

    def update(self, users=None, items=None, ratings=None):
        super().update(users, items, ratings)
        if self._user_based:
            self._feature_matrix = scipy.sparse.csr_matrix(self._ratings)
            if self._use_content:
                self._feature_matrix = scipy.sparse.hstack([self._feature_matrix, self._users])
            self._similarity_matrix = cosine_similarity(X, X, self._shrinkage)
            self._means = flatten(self._ratings.sum(axis=1)) / self._ratings.getnnz(axis=1)
        else:
            self._feature_matrix = scipy.sparse.csr_matrix(self._ratings.T)
            if self._use_content:
                self._feature_matrix = scipy.sparse.hstack([self._feature_matrix, self._items])
            self._similarity_matrix = cosine_similarity(X, X, self._shrinkage)
            self._means = flatten(self._ratings.sum(axis=0)) / self._ratings.getnnz(axis=0)

    def _predict(self, user_item):
        preds = []
        for user_id, item_id in user_item:
            if self._user_based:
                relevant_users = nlargest_indices(self._neighborhood_size,
                                                 self._similarity_matrix[user_id])
                similarities = self._similarity_matrix[relevant_users, user_id]
                ratings = flatten(self._ratings[relevant_users, item_id])
                mean = self._means[user_id]
                relevant_means = self._user_means[relevant_users]
            else:
                relevant_items = nlargest_indices(self._neighborhood_size,
                                                  self._similarity_matrix[item_id])
                similarities = self._similarity_matrix[relevant_items, item_id]
                ratings = flatten(self._ratings.T[relevant_items, user_id])
                mean = self._means[item_id]
                relevant_means = self._item_means[relevant_items]

            nonzero = ratings != 0
            ratings = ratings[nonzero]
            similarities = similarities[nonzero]
            if self._use_means:
                preds.append(mean + np.average(ratings - relevant_means[nonzero],
                                               weights=similarities))
            else:
                preds.append(np.average(ratings, weigths=similarities))

        return np.array(preds)

def cosine_similarity(X, Y, shrikage):
    return np.array(X @ Y.T / (scipy.sparse.linalg.norm(X, axis=0) *
                               scipy.sparse.linalg.norm(Y.T, axis=1) + shrinkage))

def nlargest_indices(n, iterable):
    nlargest = heapq.nlargest(self._neighborhood_size,
                              enumerate(self._similarity_matrix[user_id]),
                              key=lambda x: x[1])
    return [i[0] for i in nlargest]

def flatten(matrix):
    return matrix.toarray().flatten()
