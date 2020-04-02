"""An implementation of the SLIM recommender.

See http://glaros.dtc.umn.edu/gkhome/node/774 for details.
"""
import numpy as np
import scipy.sparse

from . import recommender


class SLIM(recommender.PredictRecommender):
    """The SLIM recommendation model which is a sparse linear method.

    Parameters
    ----------
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
                 alpha=1.0,
                 l1_ratio=0.1,
                 positive=True,
                 max_iter=100,
                 tol=1e-4,
                 seed=0):
        """Create a SLIM recommender."""
        self._model = ElasticNet(alpha=alpha,
                                 l1_ratio=l1_ratio,
                                 positive=positive,
                                 fit_intercept=False,
                                 copy_X=False,
                                 precompute=True,
                                 selection='random',
                                 max_iter=max_iter,
                                 tol=tol,
                                 random_state=seed)
        super().__init__()

    def _predict(self, user_item):  # noqa: D102
        # TODO: Implement this.
        pass
