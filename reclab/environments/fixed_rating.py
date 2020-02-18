"""A simple environment for debugging. Each user will either always rate an item a 1 or a 5."""
import numpy as np
import scipy

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
    num_init_ratings: : int
        The number of initial ratings available when the environment is reset.

    """

    def __init__(self, num_users, num_items,
                 rating_frequency=0.2, num_init_ratings=0):
        """Create a FixedRating environment."""
        super.__init__(rating_frequency, num_init_ratings)
        self._num_users = num_users
        self._num_items = num_items

    def _reset_state(self):  # noqa: D102
        self._users = [np.zeros(0)] * self._num_users
        self._items = [np.zeros(0)] * self._num_items

    def _rate_item(self, user_id, item_id):  # noqa: D102
        if item_id < self._num_items / 2:
            rating = 1.0
        else:
            rating = 5.0
        return rating
