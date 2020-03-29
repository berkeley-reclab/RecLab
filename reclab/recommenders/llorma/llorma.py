"""Tensorflow implementation of AutoRec recommender."""
import numpy as np

from .llorma_lib import llorma_g
from .. import recommender


class Llorma(recommender.PredictRecommender):
    """Many local low rank models averaged via kernels.

    Parameters
    ---------
    train.data : np.matrix
        [[user_id, item_id, rating][...]]
        Existing user_item ratings to train the recommender
    valid_data : np.matrix
        User_item_ratings matrix to validate and tune recommender
    test_data : np.matrix
        User_item_ratings matrix to test recommender
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
                 gpu_memory_frac=0.95,
                 result_path='results'):
        """Create new Local Low-Rank Matrix Approximation (LLORMA) recommender."""
        super().__init__()

        self.model = llorma_g.Llorma(n_anchor, pre_rank,
                                     pre_learning_rate, pre_lambda_val, pre_train_steps,
                                     rank, learning_rate, lambda_val, train_steps,
                                     batch_size, use_cache, gpu_memory_frac, result_path)

    def _predict(self, user_item, round_rat=False):  # noqa: W0221
        """
        Predict items for user-item pairs.

        round_rat : bool
            LLORMA treats ratings as continuous, not discrete. Set to true to round to integers.

        """
        users, items, _ = list(zip(*user_item))
        user_item = np.column_stack((users, items))
        estimate = self.model.predict(user_item)
        if round_rat:
            estimate = estimate.astype(int)
        return estimate

    def update(self, users=None, items=None, ratings=None):
        """ Update the recommender with new user, item, and rating data.

        Parameters
        ----------
        users : dict, optional
            The new users where the key is the user id while the value is the
            user features.
        items : dict, optional
            The new items where the key is the user id while the value is the
            item features.
        ratings : dict, optional
            The new ratings where the key is a double whose first index is the
            id of the user making the rating and the second index is the id of the item being
            rated. The value is a double whose first index is the rating value and the second
            index is a numpy array that represents the context in which the rating was made.
        """
        super().update(users, items, ratings)
        updated_ratings = dict(self._ratings)
        user_items = np.array(list(updated_ratings.keys()))
        rating_arr = list(updated_ratings.values())

        data = np.column_stack((user_items, rating_arr))
        self.model.reset_data(data, data, data)
        self.model.train()
