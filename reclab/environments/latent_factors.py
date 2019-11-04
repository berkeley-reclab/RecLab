import numpy as np
import scipy

from reclab.environments.simple import Simple


class LatentFactorBehavior(Simple):
    def __init__(self, latent_dim, num_users, num_items,
                 rating_frequency=0.02, num_init_ratings=0):
        self._random = np.random.RandomState()
        self._noise = 0.0
        self._latent_dim = latent_dim
        self._num_users = num_users
        self._num_items = num_items
        self._rating_frequency=rating_frequency
        self._num_init_ratings = num_init_ratings
        self._users = None
        self._items = None
        self._ratings = None

    def _init_user_item_models(self):
        # Initialization size determined such that ratings generally fall in 0-5 range
        factor_sd = np.sqrt( np.sqrt(0.5 * self._latent_dim) )
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

        self._users = (user_factors, user_bias)
        self._items = (item_factors, item_bias)
        self._offset = offset

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
        (user_factors, user_bias) = self._users
        (item_factors, item_bias) = self._items
        raw_rating = np.dot(user_factors[user_id], item_factors[item_id]) + user_bias[user_id] + item_bias[item_id] + self._offset
        rating = np.clip(raw_rating + self._random.randn() * self._noise, 0, 5).astype(np.int)
        self._ratings[user_id, item_id] = rating
        return rating
