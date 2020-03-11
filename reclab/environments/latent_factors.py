"""Contains the implementation for the Latent Behavior environment.

In this environment users and items both have latent vectors, and
the rating is determined by the inner product. Users and item both
have bias terms, and there is an underlying bias as well.
"""
import json
import os

import numpy as np

from . import environment
from .. import data_utils
from ..recommenders import LibFM


class LatentFactorBehavior(environment.DictEnvironment):
    """An environment where users and items have latent factors and biases.

    Ratings are generated as
    r = clip( <p_u, q_i> + b_u + b_i + b_0 )
    where p_u is a user's latent factor, q_i is an item's latent factor,
    b_u is a user bias, b_i is an item bias, and b_0 is a global bias.

    Parameters
    ----------
    latent_dim : int
        Size of latent factors p, q.
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
        How much the user's latent factor is shifted towards that of an item.
    memory_length : int
        The number of recent items a user remembers which affect the rating.
    boredom_threshold : int
        The size of the inner product between a new item and an item in the
        user's history to trigger a boredom response.
    boredom_penalty : float
        The factor on the penalty on the rating when a user is bored. The penalty
        is the average of the values which exceed the boredom_threshold, and the decrease
        in rating is the penalty multiplied by this factor.

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
        self._user_factors = None
        self._user_biases = None
        self._item_factors = None
        self._item_biases = None
        self._offset = None

    @property
    def name(self):
        """Name of environment, used for saving."""
        return 'latent'

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
        raw_rating = (self._user_factors[user_id] @ self._item_factors[item_id]
                      + self._user_biases[user_id] + self._item_biases[item_id] + self._offset)
        recent_item_factors = [self._item_factors[item] for item in self._user_histories[user_id]]
        boredom_penalty = 0
        for item_factor in recent_item_factors:
            if item_factor is not None:
                similarity = ((self._item_factors[item_id] @ item_factor)
                              / np.linalg.norm(item_factor)
                              / np.linalg.norm(self._item_factors[item_id]))
                if similarity > self._boredom_threshold:
                    boredom_penalty += (similarity - self._boredom_threshold)
        boredom_penalty *= self._boredom_penalty
        rating = np.clip(raw_rating - boredom_penalty + self._random.randn() * self._noise, 0, 5)
        # Updating underlying affinity
        self._user_factors[user_id] = ((1.0 - self._affinity_change) * self._user_factors[user_id]
                                       + self._affinity_change * self._item_factors[item_id])
        return rating

    def _reset_state(self):
        """Reset the state of the environment."""
        user_factors, user_bias, item_factors, item_bias, offset = self._generate_latent_factors()
        self._user_factors = user_factors
        self._user_biases = user_bias
        self._item_factors = item_factors
        self._item_biases = item_bias
        self._offset = offset

        self._users = {user_id: np.zeros(0) for user_id in range(self._num_users)}
        self._items = {item_id: np.zeros(0) for item_id in range(self._num_items)}

    def _generate_latent_factors(self):
        """Generate random latent factors."""
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


class MovieLens100k(LatentFactorBehavior):
    """An environment where user behavior is based on the ML-100k dataset.

    Latent factor model of behavior with parameters fit directly from full dataset.

    Parameters
    ----------
    latent_dim : int
        Size of latent factors p, q.
    datapath : str
        The path to the movielens datafiles and model file
    force_retrain : bool
        Forces retraining the latent factor model

    """

    def __init__(self, latent_dim, datapath, force_retrain=False,
                 **kwargs):
        """Create a ML100K Latent Factor environment."""
        self.datapath = os.path.expanduser(datapath)
        self._force_retrain = force_retrain
        num_users = 943
        num_items = 1682
        super().__init__(latent_dim, num_users, num_items, **kwargs)

    @property
    def name(self):
        """Name of environment, used for saving."""
        return 'ml100k'

    def _generate_latent_factors(self):
        """Create latent factors based on ML100K dataset."""
        model_file = os.path.join(self.datapath, 'fm_model.npz')
        if not os.path.isfile(model_file) or self._force_retrain:
            print('Did not find model file at {}, loading data for training'.format(model_file))

            users, items, ratings = data_utils.read_movielens100k()
            print('Initializing latent factor model')
            recommender = LibFM(num_user_features=0, num_item_features=0, num_rating_features=0,
                                max_num_users=self._num_users, max_num_items=self._num_items,
                                num_two_way_factors=self._latent_dim)
            recommender.reset(users, items, ratings)
            print('Training latent factor model')

            res = recommender.model_parameters()
            global_bias, weights, pairwise_interactions = res

            # TODO: this logic is only correct if there are no additional user/item/rating features
            user_indices = np.arange(self._num_users)
            item_indices = np.arange(self._num_users, self._num_users + self._num_items)

            user_factors = pairwise_interactions[user_indices]
            user_bias = weights[user_indices]
            item_factors = pairwise_interactions[item_indices]
            item_bias = weights[item_indices]
            offset = global_bias
            params = json.dumps(recommender.hyperparameters())

            np.savez(model_file, user_factors=user_factors, user_bias=user_bias,
                     item_factors=item_factors, item_bias=item_bias, offset=offset,
                     params=params)

            return user_factors, user_bias, item_factors, item_bias, offset

        model = np.load(model_file)
        print('Loading model from {} trained via:\n{}.'.format(model_file, model['params']))
        return (model['user_factors'], model['user_bias'], model['item_factors'],
                model['item_bias'], model['offset'])
