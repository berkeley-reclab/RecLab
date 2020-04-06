"""Tensorflow implementation of AutoRec recommender."""
import numpy as np

from .llorma_lib import llorma_g
from .. import recommender


class Llorma(recommender.PredictRecommender):
    """Many local low rank models averaged via kernels.

    Parameters
    ---------
    n_anchor : int
        Number of local model to build in the train phase
    pre_rank : int
        Dimension of the pre-train user/item latent factors
    rank : int
        Dimension of the train user/item factors
    pre_lambda_val : float
        Regularization parameter for the pre-train matrix factorization
    lambda_val : float
        Regularization parameter for the train model
    pre_learning_rate : float
        Learning rate when optimizing the pre-train matrix factorization
    learning_rate : float
        Learning rate for the the train model
    pre_train_steps : int
        Number of epochs in the pre-train phase
    train_steps : int
        Number of epochs in the training phase
    batch_size : int
        Batch size in training phase
    use_cache : bool
        If True use stored pre-trained item/user latent factors
    results_path :
        Folder to save model outputs and checkpoints.
    """

    def __init__(self,
                 n_anchor=10,
                 pre_rank=5,
                 pre_learning_rate=2e-4,
                 pre_lambda_val=10,
                 pre_train_steps=100,
                 rank=10,
                 learning_rate=1e-2,
                 lambda_val=1e-3,
                 train_steps=1000,
                 batch_size=128,
                 use_cache=False,
                 result_path='results'):
        """Create new Local Low-Rank Matrix Approximation (LLORMA) recommender."""
        super().__init__()

        self.model = llorma_g.Llorma(n_anchor, pre_rank,
                                     pre_learning_rate, pre_lambda_val, pre_train_steps,
                                     rank, learning_rate, lambda_val, train_steps,
                                     batch_size, use_cache, result_path)

    def _predict(self, user_item):  # noqa: W0221
        users, items, _ = list(zip(*user_item))
        users = np.array(users)
        items = np.array(items)
        # check that both the item and the user have been seen in historical data
        is_seen_uid = np.array(users <= (self.model.batch_manager.n_user - 1))
        is_seen_iid = np.array(items <= (self.model.batch_manager.n_item - 1))
        is_seen_id = np.logical_and(is_seen_iid, is_seen_uid)

        seen_user_item = np.column_stack((users[is_seen_id], items[is_seen_id]))
        seen_estimate = self.model.predict(seen_user_item)
        # choose the mean of the seen values as the estimate for the unseen ids
        unseen_estimate = np.mean(seen_estimate)
        estimate = np.ones(len(users))*unseen_estimate
        estimate[is_seen_id] = seen_estimate
        return estimate

    def update(self, users=None, items=None, ratings=None):  # noqa: W0221
        super().update(users, items, ratings)
        updated_ratings = dict(self._ratings)
        user_items = np.array(list(updated_ratings.keys()))
        rating_arr = list(updated_ratings.values())

        data = np.column_stack((user_items, rating_arr))
        self.model.reset_data(data, data, data)
        self.model.train()
