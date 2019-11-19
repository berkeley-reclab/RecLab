"""Contains the implementation for the Latent Behavior environment.

In this environment users and items both have latent vectors, and
the rating is determined by the inner product. Users and item both
have bias terms, and there is an underlying bias as well.
"""
import numpy as np
import scipy
import os
import pandas as pd

from . import environment
from reclab.recommenders.libfm.libfm import LibFM


class LatentFactorBehavior(environment.DictEnvironment):
    """An environment where users and items have latent factors and biases.

    Ratings are generated as
    r = clip( <p_u, q_i> + b_u + b_i + b_0 )

    Parameters
    ----------
    latent_dim : int
        size of latent factors p, q
    num_users : int
        The number of users in the environment.
    num_items : int
        The number of items in the environment.
    rating_frequency : float
        The proportion of users that will need a recommendation at each step.
        Must be between 0 and 1.
    num_init_ratings : int
        The number of ratings available from the start. User-item pairs are randomly selected.
    noise : float
        The standard deviation of the noise added to ratings.
    affinity_change : float
        How much the user's latent factor is shifted towards that of an item
    memory_length : int
        The number of recent items a user remembers which affect the rating
    boredom_threshold : int
        The size of inner product that a new item has to be with one in history
        to result in a boredom response
    boredom_penalty : float
        The factor on the penalty on the rating when a user is bored

    """

    def __init__(self, latent_dim, num_users, num_items,
                 rating_frequency=0.02, num_init_ratings=0,
                 noise=0.0, memory_length=0, affinity_change=0.0,
                 boredom_threshold=0, boredom_penalty=0.0):
        """Create a Latent Factor environment."""
        super().__init__(rating_frequency, num_init_ratings, memory_length)
        self._latent_dim = latent_dim
        self._num_users = num_users
        self._num_items = num_items
        self._noise = noise
        self._affinity_change = affinity_change
        self._boredom_threshold = boredom_threshold
        self._boredom_penalty = boredom_penalty
        if self._memory_length > 0:
            self._boredom_penalty /= self._memory_length
        self.name = 'latent'

    def _rate_item(self, user_id, item_id, online=True):
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
        (user_factors, user_bias) = self._users_factor_bias
        (item_factors, item_bias) = self._items_factor_bias
        raw_rating = np.dot(user_factors[user_id], item_factors[item_id]) \
            + user_bias[user_id] + item_bias[item_id] + self._offset
        boredom_penalty = 0
        for item_factor in self._user_histories[user_id]:
            if item_factor is not None:
                similarity = np.dot(item_factors[item_id], item_factor) \
                    / np.linalg.norm(item_factor) / np.linalg.norm(item_factors[item_id])
                if similarity > self._boredom_threshold:
                    boredom_penalty += (similarity-self._boredom_threshold)
        boredom_penalty *= self._boredom_penalty
        print("boredom penalty", boredom_penalty)
        rating = np.clip(raw_rating - boredom_penalty + self._random.randn() * self._noise, 0, 5)
        # Updating underlying affinity
        self._users_factor_bias[0][user_id] = (1.0 - self._affinity_change) \
            * user_factors[user_id] + self._affinity_change * item_factors[item_id]
        # Updating history
        if self._memory_length > 0 and online:
            self._user_histories[user_id] = self._user_histories[user_id][1:] \
                + [item_factors[item_id]]
        return rating

    def _reset_state(self):
        """Reset the state of the environment."""
        user_factors, user_bias, item_factors, item_bias, offset = self._generate_latent_factors()

        self._users_factor_bias = (user_factors, user_bias)
        self._items_factor_bias = (item_factors, item_bias)
        self._offset = offset

        self._users = {user_id: np.zeros(0) for user_id in range(self._num_users)}
        self._items = {item_id: np.zeros(0) for item_id in range(self._num_items)}
        self._user_histories = {user_id: [None]*self._memory_length for user_id in
                                range(self._num_users)}

    def _generate_latent_factors(self):
        # Initialization size determined such that ratings generally fall in 0-5 range
        factor_sd = np.sqrt(np.sqrt(0.5 * self._latent_dim))
        # User latent factors are normally distributed
        user_bias = np.random.normal(loc=0., scale=0.5, size=self._num_users)
        user_factors = np.random.normal(loc=0., scale=factor_sd,
                                        size=(self._num_users, self._latent_dim))
        # Item latent factors are normally distributed
        item_bias = np.random.normal(loc=0., scale=0.5, size=self._num_items)
        item_factors = np.random.normal(loc=0., scale=factor_sd,
                                        size=(self._num_items, self._latent_dim))
        # Shift up the mean
        offset = 2.5
        return user_factors, user_bias, item_factors, item_bias, offset
