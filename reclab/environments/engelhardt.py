"""
Contains the implementation for the Engelhardt environment from the algorithmic confounding paper.

In this environment users have a hidden preference for each topic and each item has a
hidden topic assigned to it.
"""

import numpy as np
import scipy
import scipy.special

from . import environment


class User:
    """Create custom User object for use in Engelhardt environment."""

    def __init__(self, num_topics, known_weight,
                 user_topic_weights, beta_var):
        """
        Initialize user with features and known/unknown utility weight.

        Each user's fraction of known utility is drawn from a beta distribution parameterized by
        a combination of the same known_weight and beta_var. known_weight and beta_var need to be
        manipulated before becoming the alpha and beta parameters to each user's distribution.

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
        self.known_weight = np.random.beta(alpha, beta)
        self.preferences = np.random.dirichlet(user_topic_weights)

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
        self.user_topic_weights = scipy.special.softmax(np.random.rand(num_topics))
        self.item_topic_weights = scipy.special.softmax(np.random.rand(num_topics))
        self._num_topics = num_topics
        self._num_users = num_users
        self._num_items = num_items
        self._users = None
        self._users_full = None
        self._items = None
        self._ratings = None

    @property
    def name(self):
        """Name of environment, used for saving."""
        return 'engelhardt'

    def _reset_state(self):
        """Reset the environment to its original state. Must be called before the first step.

        Returns
        -------
        users : np.ndarray
            This will always be an array where every row has
            size 0 since users don't have features.
        items : np.ndarray
            This will always be an array where every row has
            size 0 since items don't have features.
        ratings : np.ndarray
            The initial ratings where ratings[i, 0] corresponds to the id of the user that
            made the rating, ratings[i, 1] corresponds to the id of the item that was rated
            and ratings[i, 2] is the rating given to that item.
        util : np.ndarray
            The initial ratings where util[i, 0] corresponds to the id of the user that
            made the rating, util[i, 1] corresponds to the id of the item that was rated
            and util[i, 2] is the true utility given of the interaction.

        """
        self._users_full = {user_id: User(self._num_topics, self.known_weight,
                                          self.user_topic_weights, self.beta_var)
                            for user_id in range(self._num_users)}
        self._users = {user_id: np.zeros(0)
                       for user_id in range(self._num_users)}
        self._items = {item_id: np.zeros(0)
                       for item_id in range(self._num_items)}
        self._item_attrs = {item_id: np.random.dirichlet(self.item_topic_weights)
                            for item_id in range(self._num_items)}

    def _rate_item(self, user_id, item_id):
        """Get a user to rate an item and update the internal rating state.

        Parameters
        ----------
        user_id : int
            The id of the user making the rating.
        item_id : int
            The id of the item being rated.

        Returns
        -------
        rating : int
            The rating the item was given by the user.

        """
        item_attr = self._item_attrs[item_id]
        util, rating = self._users_full[user_id].rate(item_attr)
        print('Util is {} and rating is {}'.format(util, rating))
        return rating
