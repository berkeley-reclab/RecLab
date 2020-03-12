"""Tensorflow implementation of AutoRec recommender."""
import tensorflow as tf

from .autorec_lib import AutoRec
from .. import recommender


class Autorec(recommender.PredictRecommender):
    """Auto-encoders meet collaborative filtering.

    Parameters
    ---------
    num_users : int
        Number of users in the environment.
    num_items : int
        Number of items in the environment.
    ratings : np.matrix
        Matrix of shape (num_users, num_items) populated with user ratings.
    hidden_neuron : int
        Output dimension of hidden neuron.
    lambda_value : float
        Coefficient for regularization while training layers.
    train_epoch : int
        Number of epochs to train for each call.
    batch_size : int
        Batch size during initial training phase.
    optimizer_method : str
        Optimizer for training model; either Adam or RMSProp.
    grad_clip : bool
        Set to true to clip gradients to [-5, 5].
    base_lr : float
        Base learning rate for optimizer.
    decay_epoch_step : int
        Number of epochs before the optimizer decays the learning rate.
    random_seed : int
        Random seed to reproduce results.
    display_step : int
        Number of training steps before printing display text.

    """

    def __init__(self, num_users, num_items, ratings=None,
                 hidden_neuron=50, lambda_value=1, train_epoch=100, batch_size=100,
                 optimizer_method='Adam', grad_clip=False, base_lr=1e-4, decay_epoch_step=50,
                 random_seed=1000, display_step=1):
        """Create new Autorec recommender."""
        super().__init__()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        seen_users = set()
        seen_items = set()

        self.model = AutoRec(sess, num_users, num_items, ratings, seen_users, seen_items,
                             hidden_neuron, lambda_value, train_epoch, batch_size, optimizer_method,
                             grad_clip, base_lr, decay_epoch_step, random_seed, display_step)

    def _predict(self, user_item, round_rat=False):
        """
        Predict items for user-item pairs.

        round_rat : bool
            Autorec treats ratings as continuous, not discrete. Set to true to round to integers.

        """
        estimate = self.model.predict(user_item)
        if round_rat:
            estimate = estimate.astype(int)
        return estimate

    def reset(self, users=None, items=None, ratings=None):  # noqa: D102
        self.model.prepare_model()
        super().reset(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        super().update(users, items, ratings)
        for user_item in ratings:
            self.model.seen_users.add(user_item[0])
            self.model.seen_items.add(user_item[1])

        self.model.R = self._ratings.toarray()
        self.model.run()
