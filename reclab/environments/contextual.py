"""
Contains the implementation for the Contextual environment.

In a contextual environment, only one user is on the platform at a time.
The user has no state, and only stays for one timestep. However, each user comes
with a context that is predictive of its preferences for items.

"""
import collections

import numpy as np

from .. import data_utils
from . import environment


class Contextual(environment.DictEnvironment):
    """
    An environment that implements the contextual bandit assumption.

    Parameters
    ----------
    name: string
        The dataset to instantiate the environment with. Can be one of: 'wiki10-31k'.
    user_dist_choice : str
        The choice of user distribution for selecting online users. By default, the subset of
        online users is chosen from a uniform distribution. Currently supports normal and lognormal.

    """

    def __init__(self, name, user_dist_choice='uniform'):
        """Create a Contextual environment."""
        self._features, self._full_ratings = data_utils.read_bandit_dataset(name)
        self._curr_user = 0
        super().__init__(rating_frequency=1,
                         num_init_ratings=0,
                         memory_length=0,
                         user_dist_choice=user_dist_choice)

    @property
    def name(self):  # noqa: D102
        return 'contextual'

    def _get_dense_ratings(self):  # noqa: D102
        return self._full_ratings[:self._curr_user + 1].toarray()

    def _reset_state(self):  # noqa: D102
        self._curr_user = 0
        self._users = collections.OrderedDict([(self._curr_user, np.zeros(0))])
        self._items = collections.OrderedDict((item_id, np.zeros(0))
                                              for item_id in range(self._full_ratings.shape[1]))

    def _rate_items(self, user_id, item_ids):  # noqa: D102
        assert user_id in self._users
        assert len(item_ids) == 1
        rating = self._full_ratings[user_id, item_ids[0]]
        return np.array([rating])

    def _rating_context(self, user_id):  # noqa: D102
        return self._features[self._curr_user].toarray().flatten()

    def _update_state(self):  # noqa: D102
        self._curr_user += 1
        self._users = collections.OrderedDict([(self._curr_user, np.zeros(0))])
        return self._users.copy(), collections.OrderedDict()
