"""The implementation for a neighborhood based recommender."""
import heapq

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from . import recommender


class KNNRecommender(recommender.PredictRecommender):
    """A neighborhood based collaborative filtering algorithm.

    The class supports both user and item based collaborative filtering.

    Parameters
    ----------
    shrinkage : float
        The shrinkage parameter applied to the similarity measure.
    neighborhood_size : int
        The number of users/items to consider when estimating a rating.
    user_based : bool
        If this variable is set to true the created object will use user-based collaborative
        filtering, otherwise it will use item-based collaborative filtering.
    use_content : bool
        Whether to use the user/item features when computing the similarity measure.
    use_means : bool
        Whether to adjust the ratings based on the mean rating of each user/item.

    """

    def __init__(self, shrinkage=0, neighborhood_size=40,
                 user_based=True, use_content=True, use_means=True):
        """Create a new neighborhood recommender."""
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
        """Extends the parent method to reset to its initial state."""
        self._feature_matrix = scipy.sparse.csr_matrix((0, 0))
        self._similarity_matrix = np.empty((0, 0))
        self._means = np.empty(0)
        super().reset(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):
        """Extends the parent method to update the state."""
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
        """Implements the parent method to predict user-item ratings."""
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
    """Compute the cosine similarity between each row vector in each matrix X and Y.

    Parameters
    ----------
    X : np.matrix
        The first matrix for which to compute the cosine similarity.
    Y : np.matrix
        The second matrix for which to compute the cosine similarity.
    shrinkage : float
        The amount of shrinkage to apply to the similarity computation.

    Returns
    -------
    similarity : np.ndarray
        The similarity array between each pairs of row, where similarity[i, j]
        is the cosine similarity between X[i] and Y[j].
    """
    return (X @ Y.T).A / (scipy.sparse.linalg.norm(X, axis=1)[:, np.newaxis] *
                          scipy.sparse.linalg.norm(Y, axis=1)[np.newaxis, :] + shrinkage)


def nlargest_indices(n, iterable):
    """Given an iterable, computes the indices of the n largest items.

    Parameters
    ----------
    n : int
        How many indices to retrieve.
    iterable : iterable
        The iterable from which to compute the n largest indices.

    Returns
    -------
    largest : list of int
        The n largest indices where largest[i] is the index of the i-th largest index.
    """
    nlargest = heapq.nlargest(n, enumerate(iterable),
                              key=lambda x: x[1])
    return [i[0] for i in nlargest]


def flatten(matrix):
    """Given a matrix return a flattened numpy array."""
    return matrix.A.flatten()
