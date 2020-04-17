"""
Contains implementation for environment in "Human Interaction with Recommendation Systems."

"""

import numpy as np

from . import environment

class Schmit(environment.DictEnvironment):
    """
    Implementation of environment with static private user preferences and user-item interactions.

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

    def __init__(self, num_users, num_items, rating_frequency=0.2,
                 num_init_ratings=0, rank=10, sigma=0.2):
        """Create an environment."""
        super().__init__(rating_frequency, num_init_ratings)
        self._num_users = num_users
        self._num_items = num_items

        self.rank = rank
        self.sigma = sigma

        # constants
        self.item0 = np.random.randn(num_items, 1) / 1.5
        self.user0 = np.random.randn(num_users, 1) / 3

        # unobserved by agents
        self.U = np.random.randn(num_users, rank) / np.sqrt(self.rank)
        self.V = np.random.randn(num_items, rank) / np.sqrt(self.rank)

        # observed by agents
        self.X = np.random.randn(num_users, rank) / np.sqrt(self.rank)
        self.Y = np.random.randn(num_items, rank) / np.sqrt(self.rank)

    @property
    def name(self):
        """Name of environment, used for saving."""
        return 'schmit'

    def true_score(self, user, item):
        """
        Calculate true score.

        user : int
            User id for calculating preferences.
        item : int
            Item id.

        """

        return float(self.item0[item] + self.user0[user] + self.U[user] @ self.V[item].T)

    def value(self, user, item):
        """
        Adds private user preferences and Gaussian noise to true score.

        user : int
            User id for calculating preferences.
        item : int
            Item id.

        """
        return float(self.true_score(user, item)
                     + self.X[user] @ self.Y[item].T
                     + self._random.normal(loc=0, scale=self.sigma))

    def _reset_state(self):
        self._users = {user_id: np.zeros((0,))
                       for user_id in range(self._num_users)}
        self._items = {item_id: np.zeros((0,))
                       for item_id in range(self._num_items)}

        self.item0 = np.random.randn(self._num_items, 1) / 1.5
        self.user0 = np.random.randn(self._num_users, 1) / 3

        self.U = np.random.randn(self._num_users, self.rank) / np.sqrt(self.rank)
        self.V = np.random.randn(self._num_items, self.rank) / np.sqrt(self.rank)
        self.X = np.random.randn(self._num_users, self.rank) / np.sqrt(self.rank)
        self.Y = np.random.randn(self._num_items, self.rank) / np.sqrt(self.rank)

    def _rate_item(self, user_id, item_id):
        return self.value(user_id, item_id)
