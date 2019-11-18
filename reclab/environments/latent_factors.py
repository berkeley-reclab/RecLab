import numpy as np
import scipy
import os
import pandas as pd

from . import environment
from reclab.recommenders.libfm.libfm import LibFM

class LatentFactorBehavior(environment.DictEnvironment):
    def __init__(self, latent_dim, num_users, num_items,
                 rating_frequency=0.02, num_init_ratings=0,
                 noise=0.0, memory_length=0, affinity_change=0.0, 
                 boredom_threshold=0, boredom_penalty=0.0):
        super().__init__(rating_frequency, num_init_ratings,memory_length)
        self._latent_dim = latent_dim
        self._num_users = num_users
        self._num_items = num_items
        self._noise = noise
        self._affinity_change = affinity_change
        self._boredom_threshold = boredom_threshold
        self._boredom_penalty = boredom_penalty 
        if self._memory_length > 0: self._boredom_penalty /= self._memory_length

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
        (user_factors, user_bias) = self._users_factor_bias
        (item_factors, item_bias) = self._items_factor_bias
        raw_rating = np.dot(user_factors[user_id], item_factors[item_id]) + user_bias[user_id] + item_bias[item_id] + self._offset
        boredom_penalty = 0
        for item_factor in self._user_histories[user_id]:
            if item_factor is not None:
                similarity = np.dot(item_factors[item_id],item_factor) / np.linalg.norm(item_factor) / np.linalg.norm(item_factors[item_id])
                if similarity > self._boredom_threshold: boredom_penalty += (similarity-boredom_threshold)
        boredom_penalty *= self._boredom_penalty
        rating = np.clip(raw_rating - boredom_penalty + self._random.randn() * self._noise, 0, 5)
        # Updating underlying affinity
        self._users_factor_bias[0][user_id] = (1.0 - self._affinity_change) * user_factors[user_id] + self._affinity_change * item_factors[item_id]
        # Updating history
        if self._memory_length > 0:
            self._user_histories[user_id] = self._user_histories[user_id][1:]+[item_factors[item_id]]
        return rating

    def _reset_state(self):
        """Reset the state of the environment."""

        user_factors, user_bias, item_factors, item_bias, offset = self._generate_latent_factors()

        self._users_factor_bias = (user_factors, user_bias)
        self._items_factor_bias = (item_factors, item_bias)
        self._offset = offset

        self._users = {user_id: np.zeros(0) for user_id in range(self._num_users)}
        self._items = {item_id: np.zeros(0) for item_id in range(self._num_items)}
        self._user_histories = {user_id: [None]*self._memory_length for user_id in range(self._num_users)}

    def _generate_latent_factors(self):
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
        return user_factors, user_bias, item_factors, item_bias, offset

class MovieLens100k(LatentFactorBehavior):
    def __init__(self, latent_dim, datapath,
                 rating_frequency=0.02, num_init_ratings=0):
        self.datapath = os.path.expanduser(datapath)
        # TODO: this should not be hardcoded
        num_users = 943
        num_items = 1682
        super().__init__(latent_dim, num_users, num_items,
                 rating_frequency, num_init_ratings)

    def _generate_latent_factors(self):
        users, items, ratings = self._read_datafile()
        recommender = LibFM(num_user_features=0, num_item_features=0, num_rating_features=0, 
                            max_num_users=self._num_users, max_num_items=self._num_items)
        recommender.init(users, items, ratings)
        global_bias, weights, pairwise_interactions = recommender.train()
        print(global_bias)
        print(weights)
        print(pairwise_interactions)
        # TODO: need to read these models out of LIBFM
        assert False
        self._users = (user_factors, user_bias)
        self._items = (item_factors, item_bias)
        self._offset = offset

    def _read_datafile(self):
        datafile = os.path.join(self.datapath, "u.data")
        if not os.path.isfile(datafile):
            raise OSError("Datafile u.data not found in {}. Download from https://grouplens.org/datasets/movielens/100k/ and follow README instructions for unzipping.".format(datafile))
        
        data = pd.read_csv(datafile, sep='\t', header=None, 
                            names=["user_id","item_id", "rating"], usecols=[0,1,2])
        # shifting user and movie indexing
        data["user_id"] -= 1
        data["item_id"] -= 1
        # validating data assumptions
        assert len(data) == 100000
        assert len(np.unique(data["user_id"])) == self._num_users
        assert len(np.unique(data["item_id"])) == self._num_items

        users = {}
        for i in range(self._num_users):
            users[i] = np.zeros((0))

        items = {}
        for i in range(self._num_items):
            items[i] = np.zeros((0))

        # Fill the rating array with initial data.
        ratings = np.array(data)
        return users, items, ratings
