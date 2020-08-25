"""A simple environment for debugging. Each user will either always rate an item a 1 or a 5."""
import numpy as np

from . import environment


class FixedRating(environment.DictEnvironment):
    """An environment in which half the users rate all items with a 1 and the other half with a 5.

    Parameters
    ----------
    num_users : int
        The number of users in the environment.
    num_items : int
        The number of items in the environment.
    rating_frequency : float
        What proportion of users will need a recommendation at each step.
    num_init_ratings : int
        The number of initial ratings available when the environment is reset.

    """

    def __init__(self, num_users, num_items,
                 rating_frequency=0.2, num_init_ratings=0):
        """Create a FixedRating environment."""
        super().__init__(rating_frequency, num_init_ratings)
        self._num_users = num_users
        self._num_items = num_items

    @property
    def name(self):  # noqa: D102
        return 'fixed'

    def _get_dense_ratings(self):  # noqa: D102
        ratings = np.ones([self._num_users, self._num_items])
        ratings[:, self._num_items // 2:] = 5.0
        return ratings

    def _reset_state(self):  # noqa: D102
        self._users = {user_id: np.zeros((0,)) for user_id in range(self._num_users)}
        self._items = {item_id: np.zeros((0,)) for item_id in range(self._num_items)}

    def _rate_items(self, user_id, item_ids):  # noqa: D102
        # Find the largest item id that has not yet been rated.
        max_id = None
        for item_id in sorted(item_ids, reverse=True):
            if (user_id, item_id) not in self._ratings:
                max_id = item_id
                break

        # If we have found an unrated item, rate it either 1 or 5.
        ratings = np.ones(len(item_ids)) * np.nan
        if max_id is not None:
            if max_id >= self._num_items // 2:
                ratings[item_ids == max_id] = 5.0
            else:
                ratings[item_ids == max_id] = 1.0

        return ratings
