"""Contains the implementation for the Latent Behavior environment.

In this environment users and items both have latent vectors, and
the rating is determined by the inner product. Users and item both
have bias terms, and there is an underlying bias as well.
"""
import collections
import json
import os

import numpy as np

from . import environment
from .. import data_utils


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
    user_dist_choice : str
        The choice of user distribution for selecting online users. By default, the subset of
        online users is chosen from a uniform distribution. Currently supports normal and lognormal.

    """

    def __init__(self, latent_dim, num_users, num_items,
                 rating_frequency=0.02, num_init_ratings=0,
                 noise=0.0, memory_length=0, affinity_change=0.0,
                 boredom_threshold=0, boredom_penalty=0.0, user_dist_choice='uniform'):
        """Create a Latent Factor environment."""
        super().__init__(rating_frequency, num_init_ratings, memory_length, user_dist_choice)
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

    def _get_dense_ratings(self):  # noqa: D102
        ratings = (self._user_factors @ self._item_factors.T + self._user_biases[:, np.newaxis] +
                   self._item_biases[np.newaxis, :] + self._offset)
        # Compute the boredom penalties.
        item_norms = np.linalg.norm(self._item_factors, axis=1)
        normalized_items = self._item_factors / item_norms[:, np.newaxis]
        similarities = normalized_items @ normalized_items.T
        similarities -= self._boredom_threshold
        similarities[similarities < 0] = 0
        penalties = self._boredom_penalty * similarities
        for user_id in range(self._num_users):
            for item_id in self._user_histories[user_id]:
                if item_id is not None:
                    ratings[user_id] -= penalties[item_id]

        return ratings

    def _get_rating(self, user_id, item_id):
        """Compute user's rating of item based on model.

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

        # Compute the boredom penalty.
        boredom_penalty = 0
        for item_id_hist in self._user_histories[user_id]:
            item_factor = self._item_factors[item_id_hist]
            if item_factor is not None:
                similarity = ((self._item_factors[item_id] @ item_factor)
                              / np.linalg.norm(item_factor)
                              / np.linalg.norm(self._item_factors[item_id]))
                if similarity > self._boredom_threshold:
                    boredom_penalty += (similarity - self._boredom_threshold)
        boredom_penalty *= self._boredom_penalty
        rating = np.clip(raw_rating - boredom_penalty + self._dynamics_random.randn() *
                         self._noise, 1, 5)

        return rating

    def _rate_items(self, user_id, item_ids):
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
        # TODO: Add support for slates of size greater than 1.
        item_id = item_ids[0]
        rating = self._get_rating(user_id, item_id)

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

        self._users = collections.OrderedDict((user_id, np.zeros(0))
                                              for user_id in range(self._num_users))
        self._items = collections.OrderedDict((item_id, np.zeros(0))
                                              for item_id in range(self._num_items))

    def _generate_latent_factors(self):
        """Generate random latent factors."""
        # Initialization size determined such that ratings generally fall in 0-5 range
        factor_sd = np.sqrt(np.sqrt(0.5 / self._latent_dim))
        # User latent factors are normally distributed
        user_bias = self._init_random.normal(loc=0., scale=0.5, size=self._num_users)
        user_factors = self._init_random.normal(loc=0., scale=factor_sd,
                                                size=(self._num_users, self._latent_dim))
        # Item latent factors are normally distributed
        item_bias = self._init_random.normal(loc=0., scale=0.5, size=self._num_items)
        item_factors = self._init_random.normal(loc=0., scale=factor_sd,
                                                size=(self._num_items, self._latent_dim))
        # Shift up the mean
        offset = 3.0
        return user_factors, user_bias, item_factors, item_bias, offset


class DatasetLatentFactor(LatentFactorBehavior):
    """An environment where user behavior is based on a dataset.

    Latent factor model of behavior with parameters fit directly from full dataset.

    Parameters
    ----------
    name : str
        The name of the dataset. Must be one of: 'ml-100k', 'ml-10m', 'lastfm'.
    latent_dim : int
        Size of latent factors p, q.
    datapath : str
        The path to the directory containing datafiles
    force_retrain : bool
        Forces retraining the latent factor model
    max_num_users : int
        The maximum number of users for the environment, if not the number in the dataset.
    max_num_items : int
        The maximum number of items for the environment, if not the number in the dataset.

    """

    def __init__(self, name, latent_dim=128, datapath=data_utils.DATA_DIR, force_retrain=False,
                 max_num_users=np.inf, max_num_items=np.inf, **kwargs):
        """Create a ML100K Latent Factor environment."""
        self.dataset_name = name
        if name == 'ml-100k':
            self.datapath = os.path.expanduser(os.path.join(datapath, 'ml-100k'))
            latent_dim = 100 if latent_dim is None else latent_dim
            self._full_num_users = 943
            self._full_num_items = 1682
            # These parameters are the result of tuning.
            reg = 0.1
            learn_rate = 0.005
            self.train_params = dict(bias_reg=reg, one_way_reg=reg, two_way_reg=reg,
                                     learning_rate=learn_rate, num_iter=100)
        elif name == 'ml-10m':
            self.datapath = os.path.expanduser(os.path.join(datapath, 'ml-10M100K'))
            latent_dim = 128 if latent_dim is None else latent_dim
            self._full_num_users = 69878
            self._full_num_items = 10677
            # these parameters are presented in "On the Difficulty of Baselines" by Rendle et al.
            reg = 0.04
            learn_rate = 0.003
            self.train_params = dict(bias_reg=reg, one_way_reg=reg, two_way_reg=reg,
                                     learning_rate=learn_rate, num_iter=128)
        elif name == 'lastfm':
            self.datapath = os.path.expanduser(os.path.join(datapath, 'lastfm-dataset-1K'))
            latent_dim = 128 if latent_dim is None else latent_dim
            self._full_num_users = 992
            self._full_num_items = 177023
            # These parameters are presented in "Recommendations and User Agency" by Dean et al.
            reg = 0.08
            learn_rate = 0.001
            self.train_params = dict(bias_reg=reg, one_way_reg=reg, two_way_reg=reg,
                                     learning_rate=learn_rate, num_iter=128)
        else:
            raise ValueError('dataset name not recognized')
        self._force_retrain = force_retrain

        num_users = min(self._full_num_users, max_num_users)
        num_items = min(self._full_num_items, max_num_items)

        super().__init__(latent_dim, num_users, num_items, **kwargs)

    @property
    def name(self):
        """Name of environment, used for saving."""
        return 'latent-{}'.format(self.dataset_name)

    def _generate_latent_factors(self):
        full_model_params = dict(num_user_features=0, num_item_features=0, num_rating_features=0,
                                 max_num_users=self._full_num_users,
                                 max_num_items=self._full_num_items,
                                 num_two_way_factors=self._latent_dim, **self.train_params)

        model_file = os.path.join(self.datapath + '-model', 'fm_model.npz')
        res = load_latent_factors(model_file)
        if res is None or self._force_retrain:
            print('Training model from scratch, either due to force_retrain flag or')
            print('\tdid not find model file at {}'.format(model_file))
            res = generate_latent_factors_from_data(self.dataset_name, model_file,
                                                    full_model_params)
            user_factors, user_bias, item_factors, item_bias, offset = res
        else:
            user_factors, user_bias, item_factors, item_bias, offset = res

        if self._num_users < self._full_num_users or self._num_items < self._full_num_items:
            num_users, num_items = (min(self._num_users, self._full_num_users),
                                    min(self._num_items, self._full_num_items))
            # TODO: may want to reduce the number in some other way
            # e.g. related to popularity
            user_indices = self._init_random.choice(user_factors.shape[0], size=num_users,
                                                    replace=False)
            item_indices = self._init_random.choice(item_factors.shape[0], size=num_items,
                                                    replace=False)
            user_factors = user_factors[user_indices]
            user_bias = user_bias[user_indices]
            item_factors = item_factors[item_indices]
            item_bias = item_bias[item_indices]
        return user_factors, user_bias, item_factors, item_bias, offset


def load_latent_factors(model_file):
    """Load pretrained latent factor model."""
    if not os.path.isfile(model_file):
        return None
    model = np.load(model_file)
    print('Loading model from {} trained via:\n{}.'.format(model_file, model['params']))

    user_factors = model['user_factors']
    user_bias = model['user_bias']
    item_factors = model['item_factors']
    item_bias = model['item_bias']
    offset = model['offset']

    return user_factors, user_bias, item_factors, item_bias, offset


def generate_latent_factors_from_data(dataset_name, model_file, params):
    """Create latent factors based on a dataset."""
    from ..recommenders import LibFM

    users, items, ratings = data_utils.read_dataset(dataset_name)
    print('Initializing latent factor model')
    recommender = LibFM(**params)
    recommender.reset(users, items, ratings)
    print('Training latent factor model with parameters: {}'.format(params))

    global_bias, weights, pairwise_interactions = recommender.model_parameters()
    if len(weights) == 0:
        weights = np.zeros(pairwise_interactions.shape[0])

    # TODO: this logic is only correct if there are no additional user/item/rating features
    # Note that we discard the original data's user_ids and item_ids at this step
    user_indices = np.arange(params['max_num_users'])
    item_indices = np.arange(params['max_num_users'],
                             params['max_num_users'] + params['max_num_items'])

    user_factors = pairwise_interactions[user_indices]
    user_bias = weights[user_indices]
    item_factors = pairwise_interactions[item_indices]
    item_bias = weights[item_indices]
    offset = global_bias
    params = json.dumps(recommender.hyperparameters)

    np.savez(model_file, user_factors=user_factors, user_bias=user_bias,
             item_factors=item_factors, item_bias=item_bias, offset=offset,
             params=params)

    return user_factors, user_bias, item_factors, item_bias, offset
