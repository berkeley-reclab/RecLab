"""
Contains the implementation for the Engelhardt environment from the algorithmic confounding paper.

In this environment users have a hidden preference for each topic and each item has a
hidden topic assigned to it.
"""
import collections

import numpy as np
import scipy
import scipy.special

from . import environment


class User:
    """Create custom User object for use in Engelhardt environment."""

    def __init__(self, num_topics, known_weight,
                 user_topic_weights, beta_var, random):
        """
        Initialize user with features and known/unknown utility weight.

        Each user's fraction of known utility is drawn from a beta distribution parameterized by
        a combination of the same known_weight and beta_var. known_weight
        and beta_var need to be manipulated
        before becoming the alpha and beta parameters to each user's distribution.

        Parameters
        ----------
        known_weight : float
            the average fraction of the user's true utility known to the user and the recommender

        user_topic_weights : float array
            global parameters for each user's topic preferences

        beta_var : int
            variance of beta distribution

        """
        self.num_topics = num_topics
        alpha = ((1 - known_weight) / (beta_var ** 2) - (1 / known_weight)) * (known_weight ** 2)
        beta = alpha * ((1 / known_weight) - 1)
        self.known_weight = random.beta(alpha, beta)
        self.preferences = random.dirichlet(user_topic_weights)

    def rate(self, item_attributes):
        """
        Return true utility and known utility from user.

        Returns
        ------
        true_util : float
            User's actual utility, including both known and unknown fractions

        known_util: float
            User's known utility, found by multiplying true utility
            by the fraction of utility that is known

        """
        true_util = np.dot(self.preferences, item_attributes) * 5
        return true_util, true_util * self.known_weight


class Engelhardt(environment.DictEnvironment):
    """
    Implementation of environment with known and unknown user utility, static over time.

    Based on "How Algorithmic Confounding in Recommendation Systems Increases Homogeneity
    and Decreases Utility" by Chaney, Stewart, and Engelhardt (2018).

    """

    def __init__(self, num_topics, num_users, num_items, rating_frequency=0.2,
                 num_init_ratings=0, known_weight=0.98, beta_var=10 ** -5):
        """Create an Engelhardt environment."""
        super().__init__(rating_frequency, num_init_ratings)
        self.known_weight = known_weight
        self.beta_var = beta_var
        self.user_topic_weights = scipy.special.softmax(self._init_random.rand(num_topics))
        self.item_topic_weights = scipy.special.softmax(self._init_random.rand(num_topics))
        self._num_topics = num_topics
        self._num_users = num_users
        self._num_items = num_items
        self._users = None
        self._users_full = None
        self._items = None
        self._ratings = None
        self._item_attrs = None

    @property
    def name(self):  # noqa: D102
        return 'engelhardt'

    def _get_dense_ratings(self):  # noqa: D102
        ratings = np.zeros([self._num_users, self._num_items])
        for user_id in range(self._num_users):
            for item_id in range(self._num_items):
                item_attr = self._item_attrs[item_id]
                ratings[user_id, item_id] = self._users_full[user_id].rate(item_attr)[1]
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

    def _rate_item(self, user_id, item_id):  # noqa: D102
        item_attr = self._item_attrs[item_id]
        _, rating = self._users_full[user_id].rate(item_attr)
        return rating
