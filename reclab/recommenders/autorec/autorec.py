import os
import tensorflow as tf
import argparse
import scipy

import numpy as np
from .autorec_lib.AutoRec import AutoRec
from .. import recommender

class Autorec(recommender.PredictRecommender):
    """
    Auto-encoders meet collaborative filtering.

    Parameters
    ---------
    num

    """
    def __init__(self, ratings, num_users, num_items,
            hidden_neuron=500, lambda_value=1, train_epoch=2000, batch_size=100,
            optimizer_method='Adam', grad_clip=False, base_lr=1e-3, decay_epoch_step=50, random_seed=1000, display_step=1):
        super().__init__()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        seen_users = set()
        seen_items = set()

        self.model = AutoRec(sess, num_users, num_items, ratings, seen_users, seen_items,
                hidden_neuron, lambda_value, train_epoch, batch_size, optimizer_method, grad_clip,
                base_lr, decay_epoch_step, random_seed, display_step)

    def _predict(self, user_item, round_rat=False):
        estimate = self.model.predict(user_item)
        if round_rat:
            estimate = estimate.astype(int)
        return estimate

    def reset(self, users=None, items=None, ratings=None):
        rating_matrix = np.zeros(shape=(len(users), len(items)))
        for user_item in ratings:
            rating_matrix[user_item[0]][user_item[1]] = ratings[user_item][0]

        self.model.R = rating_matrix
        self.model.seen_users = set(users)
        self.model.seen_items = set(items)
        self.model.run()
        super().reset(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):
        super().update(users, items, ratings)
        for user in users:
            self.model.seen_users.add(user)
        for item in items:
            self.model.seen_items.add(item)
        for user_item in ratings:
            self.model.R[user_item[0]][user_item[1]] = ratings[user_item][0]
        self.model.train_model(0)
