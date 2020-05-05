"""An implementation of SLIM and EASE sparse linear recommenders.

For details, see:
  - http://glaros.dtc.umn.edu/gkhome/node/774
  - https://arxiv.org/pdf/1905.03375.pdf
"""
import warnings
import numpy as np
import scipy.sparse
import sklearn.linear_model
from sklearn.exceptions import ConvergenceWarning

from . import recommender

warnings.simplefilter('ignore', category=ConvergenceWarning)


class SLIM(recommender.PredictRecommender):
    """The SLIM recommendation model which is a sparse linear method.

    Parameters
    ----------
    binarize : boolean
        Determines whether to binarize ratings before fitting a model.
    alpha : float
        Constant that multiplies the regularization terms.
    l1_ratio : float
        The ratio of the L1 regularization term with respect to the L2 regularization.
    max_iter : int
        The maximum number of iterations to train the model for.
    tol : float
        The tolerance below which the optimization will stop.
    seed : int
        The random seed to use when training the model.

    """

    def __init__(self,
                 binarize=False,
                 alpha=1.0,
                 l1_ratio=0.1,
                 positive=True,
                 max_iter=100,
                 tol=1e-4,
                 seed=0):
        """Create a SLIM recommender."""
        super().__init__()
        self._binarize = binarize
        self._model = sklearn.linear_model.ElasticNet(alpha=alpha,
                                                      l1_ratio=l1_ratio,
                                                      positive=positive,
                                                      fit_intercept=False,
                                                      copy_X=False,
                                                      precompute=True,
                                                      selection='random',
                                                      max_iter=max_iter,
                                                      tol=tol,
                                                      random_state=seed)
        self._weights = None
        self._hyperparameters.update(locals())

        # We only want the function arguments so remove class related objects.
        del self._hyperparameters['self']
        del self._hyperparameters['__class__']

    @property
    def name(self):  # noqa: D102
        return 'slim'

    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        super().update(users, items, ratings)
        num_items = len(self._items)
        self._weights = scipy.sparse.dok_matrix((num_items, num_items))
        if self._binarize:
            row, col = self._ratings.nonzero()
            data = np.ones(len(row))
            ratings = scipy.sparse.csr_matrix((data, (row, col)), shape=self._ratings.shape).tolil()
        else:
            ratings = self._ratings.tolil()
        for item_id in range(num_items):
            target = ratings[:, item_id].toarray()
            # Zero out the column of the current item to prevent a trivial solution.
            ratings[:, item_id] = 0
            # Fit the mode and save the weights
            # This currently takes 0.02s/item on ML100k
            self._model.fit(ratings, target)
            self._weights[:, item_id] = self._model.sparse_coef_.T
            self._weights[item_id, item_id] = 0
            # Restore the rating column.
            ratings[:, item_id] = target
        self._weights = scipy.sparse.csr_matrix(self._weights)

    @property
    def dense_predictions(self):  # noqa: D102
        if self._dense_predictions is None:
            self._dense_predictions = (self._ratings @ self._weights).todense()
        return self._dense_predictions

    def _predict(self, user_item):  # noqa: D102
        # Predict on all user-item pairs.
        all_predictions = self.dense_predictions
        predictions = []
        for user_id, item_id, _ in user_item:
            predictions.append(all_predictions[user_id, item_id])

        return np.array(predictions)


class EASE(recommender.PredictRecommender):
    """The EASE recommendation model which is a simple linear method.

    Parameters
    ----------
    binarize : boolean
        Determines whether to binarize ratings before fitting a model.
    lam : float
        Constant that multiplies the regularization terms.

    """

    def __init__(self,
                 binarize=False,
                 lam=1.0):
        """Create an EASE recommender."""
        super().__init__()

        self._binarize = binarize
        self._lam = lam

        self._weights = None
        self._hyperparameters.update(locals())

        # We only want the function arguments so remove class related objects.
        del self._hyperparameters['self']
        del self._hyperparameters['__class__']

    @property
    def name(self):  # noqa: D102
        return 'ease'

    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        super().update(users, items, ratings)

        if self._binarize:
            row, col = self._ratings.nonzero()
            data = np.ones(len(row))
            ratings = scipy.sparse.csr_matrix((data, (row, col)), shape=self._ratings.shape)
        else:
            ratings = self._ratings

        item_products = ratings.T @ ratings

        diag_ind = np.diag_indices(item_products.shape[0])
        item_products[diag_ind] += self._lam
        inverse_mat = np.linalg.inv(item_products.todense())
        self._weights = inverse_mat / (-np.diag(inverse_mat))
        self._weights[diag_ind] = 0

    @property
    def dense_predictions(self):  # noqa: D102
        if self._dense_predictions is None:
            self._dense_predictions = (self._ratings @ self._weights)
        return self._dense_predictions

    def _predict(self, user_item):  # noqa: D102
        # Predict on all user-item pairs.
        all_predictions = self.dense_predictions
        predictions = []
        for user_id, item_id, _ in user_item:
            predictions.append(all_predictions[user_id, item_id])

        return np.array(predictions)
