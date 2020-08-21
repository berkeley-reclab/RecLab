"""
Contains the implementation for the BetaRank environment from the algorithmic confounding paper.

In this environment users have a hidden preference for each topic and each item has a
hidden topic assigned to it.
"""
import collections

import numpy as np
import scipy
import scipy.special

from . import environment


class BetaRank(environment.DictEnvironment):
    """
    Implementation of environment with known and unknown user utility, static over time.

    Based on "How Algorithmic Confounding in Recommendation Systems Increases Homogeneity
    and Decreases Utility" by Chaney, Stewart, and Engelhardt (2018).

    """

    def __init__(self, dimension, num_users, num_items, rating_frequency=0.2,
                 num_init_ratings=0, known_mean=0.98, user_dist_choice='uniform'):
        """Create a BetaRank environment."""
        super().__init__(rating_frequency, num_init_ratings, 0, user_dist_choice)
        self._user_preferences = self._init_random.dirichlet(
            10 * self._init_random.dirichlet(np.ones(dimension)),
            size=num_users
        )
        self._item_preferences = self._init_random.dirichlet(
            0.1 * self._init_random.dirichlet(100 * np.ones(dimension)),
            size=num_items
        )
        self._known_mean = known_mean

    @property
    def name(self):  # noqa: D102
        return 'beta-rank'

    def _get_dense_ratings(self):  # noqa: D102
        ratings = np.zeros([self._num_users, self._num_items])
        for user_id in range(self._num_users):
            for item_id in range(self._num_items):
                item_attr = self._item_attrs[item_id]
                ratings[user_id, item_id] = self._users_full[user_id].rate(item_attr)
        return ratings

    def _reset_state(self):  # noqa: D102
        self._users_full = {user_id: User(self._num_topics, self.known_weight,
                                          self.user_topic_weights, self.beta_var,
                                          self._init_random)
                            for user_id in range(self._num_users)}
        self._item_attrs = {item_id: self._init_random.dirichlet(self.item_topic_weights)
                            for item_id in range(self._num_items)}
        self._users = collections.OrderedDict((user_id, np.zeros(0))
                                              for user_id in range(self._num_users))
        self._items = collections.OrderedDict((item_id, np.zeros(0))
                                              for item_id in range(self._num_items))

    def _rate_items(self, user_id, item_ids):  # noqa: D102
        means = self._item_preferences[item_ids] @ self._user_preferences[user_id]
        values = self._beta_prime(means)
        known = self._beta_prime(self._known_mean, shape=len(item_ids))
        chosen_idx = np.argmax(np.arange(1, len(item_ids) + 1) ** (-0.8) * known * values)
        ratings = np.ones(len(item_ids)) * np.nan
        ratings[chosen_idx] = values[chosen_idx]
        return ratings

    def _beta_prime(mean, std_dev=1e-5):
        alpha = ((1 - mean) / std_dev ** 2 - 1 / mean) * mean ** 2
        beta = alpha * (1 / mean - 1)
        return self._dynamics_random.beta(alpha, beta)
