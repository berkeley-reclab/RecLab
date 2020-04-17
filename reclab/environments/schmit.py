import collections
import functools as ft
import math
import json
import random

import numpy as np
import scipy as sp
import scipy.linalg

from . import environment


class Schmit(environment.DictEnvironment):
    """
    Implementation of environment with static private user preferences and
    defined user-item interactions.

    Based on "Human Interaction with Recommendation Systems" by Schmit and Riquelme (2018).

    Parameters
    ----------
    _num_users : int
        The number of users in the environment.
    _num_items : int
        The number of items in the environment.
    rating_frequency : float
        What proportion of users will need a recommendation at each step.
    num_init_ratings: : int
        The number of initial ratings available when the environment is reset.
    rank : int
        Rank of user preferences.
    sigma : float
        Variance of the Gaussian noise added to determine user-item value.

    """

    def __init__(self, _num_users, _num_items, rating_frequency=0.2,
                 num_init_ratings=0, rank=10, sigma=0.2):
        super().__init__(rating_frequency, num_init_ratings)
        self._num_users = _num_users
        self._num_items = _num_items

        self.rank = rank
        self.sigma = sigma

        alpha_rank = 10
        nobs_user = int(alpha_rank * rank)
        perc_data = nobs_user / _num_items

        # constants
        self.item0 = np.random.randn(_num_items, 1) / 1.5
        self.user0 = np.random.randn(_num_users, 1) / 3

        # unobserved by agents
        self.U = np.random.randn(_num_users, rank) / np.sqrt(self.rank)
        self.V = np.random.randn(_num_items, rank) / np.sqrt(self.rank)

        # observed by agents
        self.X = np.random.randn(_num_users, rank) / np.sqrt(self.rank)
        self.Y = np.random.randn(_num_items, rank) / np.sqrt(self.rank)

    @property
    def name(self):
        """Name of environment, used for saving."""
        return 'schmit'

    def true_score(self, user, item):
        """
        Model:
        V_ij = a_i + b_j + u_i @ v_j + x_i @ y_j + eps
        where x_i @ y_j is the private information known to the user.

        """

        return float(self.item0[item] + self.user0[user] + self.U[user] @ self.V[item].T)

    def value(self, user, item):
        return float(self.true_score(user, item) + self.X[user] @ self.Y[item].T + random.gauss(0, self.sigma))

    def unbiased_value(self, user, item):
        return self.true_score(user, item) + random.gauss(0, self.sigma)

    def _reset_state(self):
        self._users = {user_id: np.zeros((0,))
                       for user_id in range(self._num_users)}
        self._items = {item_id: np.zeros((0,))
                       for item_id in range(self._num_items)}

        self.item0 = np.random.randn(self._num_items, 1) / 1.5
        self.user0 = np.random.randn(self._num_users, 1) / 3

        self.U = np.random.randn(self._num_users, self.rank) / np.sqrt(self.rank)
        self.V = np.random.randn(self._num_items, self.rank) / np.sqrt(self.rank)

        self.U = np.random.randn(self._num_users, self.rank) / np.sqrt(self.rank)
        self.V = np.random.randn(self._num_items, self.rank) / np.sqrt(self.rank)

    def _rate_item(self, user_id, item_id):
        return self.value(user_id, item_id)
