"""
Contains implementation for environment in "Human Interaction with Recommendation Systems".

https://arxiv.org/pdf/1703.00535.pdf
"""

import numpy as np

from . import environment


class Schmit(environment.DictEnvironment):
    """
    Implementation of environment with static private user preferences and user-item interactions.

    Based on "Human Interaction with Recommendation Systems" by Schmit and Riquelme (2018).

    Parameters
    ----------
    num_users : int
        The number of users in the environment.
    num_items : int
        The number of items in the environment.
    rating_frequency : float
        What proportion of users will need a recommendation at each step.
    num_init_ratings: : int
        The number of initial ratings available when the environment is reset.
    rank : int
        Rank of user preferences.
    sigma : float
        Variance of the Gaussian noise added to determine user-item value.
    user_dist_choice : str
        The choice of user distribution for selecting online users. By default, the subset of
        online users is chosen from a uniform distribution. Currently supports normal and lognormal.

    """

    def __init__(self, num_users, num_items, rating_frequency=0.2,
                 num_init_ratings=0, rank=10, sigma=0.2,
                 user_dist_choice='uniform'):
        """Create an environment."""
        super().__init__(rating_frequency, num_init_ratings, 0, user_dist_choice)
        self._num_users = num_users
        self._num_items = num_items

        self.rank = rank
        self.sigma = sigma

        # constants
        self.item_bias = self._init_random.randn(num_items, 1) / 1.5
        self.user_bias = self._init_random.randn(num_users, 1) / 3

        # unobserved by agents
        self.U = self._init_random.randn(num_users, rank) / np.sqrt(self.rank)
        self.V = self._init_random.randn(num_items, rank) / np.sqrt(self.rank)

        # observed by agents
        self.X = self._init_random.randn(num_users, rank) / np.sqrt(self.rank)
        self.Y = self._init_random.randn(num_items, rank) / np.sqrt(self.rank)

    @property
    def name(self):
        """Name of environment, used for saving."""
        return 'schmit'

    def true_score(self, user, item):
        """
        Calculate true score.

        Parameters
        ----------
        user : int
            User id for calculating preferences.
        item : int
            Item id.

        Returns
        -------
        score : float
            The true score of the item for the user.

        """
        return float(self.item_bias[item] + self.user_bias[user] + self.U[user] @ self.V[item].T)

    def value(self, user, item):
        """
        Add private user preferences and Gaussian noise to true score.

        Parameters
        ----------
        user : int
            User id for calculating preferences.
        item : int
            Item id.

        Returns
        -------
        value : float
            The (noisy) value of the item to the user.

        """
        ratings = float(self.true_score(user, item) + self.X[user] @ self.Y[item].T +
                        self._dynamics_random.normal(loc=0, scale=self.sigma) + 3)
        return np.clip(ratings, 1, 5)

    def _reset_state(self):
        self._users = {user_id: np.zeros((0,))
                       for user_id in range(self._num_users)}
        self._items = {item_id: np.zeros((0,))
                       for item_id in range(self._num_items)}

        self.item_bias = self._init_random.randn(self._num_items, 1) / 1.5
        self.user_bias = self._init_random.randn(self._num_users, 1) / 3

        self.U = self._init_random.randn(self._num_users, self.rank) / np.sqrt(self.rank)
        self.V = self._init_random.randn(self._num_items, self.rank) / np.sqrt(self.rank)
        self.X = self._init_random.randn(self._num_users, self.rank) / np.sqrt(self.rank)
        self.Y = self._init_random.randn(self._num_items, self.rank) / np.sqrt(self.rank)

    def _rate_item(self, user_id, item_id):
        return self.value(user_id, item_id)

    def _get_dense_ratings(self):
        """Compute all the true ratings on every user-item pair at the current timestep.

        A true rating is defined as the rating a user would make with all noise removed.

        Returns
        -------
        dense_ratings : np.ndarray
            The array of all true ratings where true_ratings[i, j] is the rating by user i
            on item j.

        """
        dense_ratings = np.zeros([self._num_users, self._num_items])
        for u in range(self._num_users):
            for i in range(self._num_items):
                dense_ratings[u, i] = self.true_score(u, i) + self.X[u] @ self.Y[i].T + 3
        return dense_ratings
