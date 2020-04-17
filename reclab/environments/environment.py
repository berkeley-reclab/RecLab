"""Defines a set of base classes from which environments can inherit.

Environment is the interface all environments must implement. The other classes represent
specific environment variants that occur often enough to be abstract classes to inherit from.
"""
import abc
import collections

import numpy as np


class Environment(abc.ABC):
    """The interface all environments must implement."""

    @abc.abstractmethod
    def reset(self):
        """Reset the environment to its original state. Must be called before the first step.

        Returns
        -------
        users : iterable
            New users and users whose information got updated this timestep.
        items : iterable
            New items and items whose information got updated this timestep.
        ratings : iterable
            New ratings and ratings whose information got updated this timestep.
        info : dict
            Extra information that can be used for debugging but should not be made accessible to
            the recommender.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, recommendations):
        """Run one timestep of the environment.

        Parameters
        ----------
        recommendations : iterable
            The recommendations made to each user. The i-th recommendation should correspond to
            the i-th user that was online at this timestep.

        Returns
        -------
        users : iterable
            New users and users whose information got updated this timestep.
        items : iterable
            New items and items whose information got updated this timestep.
        ratings : iterable
            New ratings and ratings whose information got updated this timestep.
        info : dict
            Extra information that can be used for debugging but should not be made accessible to
            the recommender.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def online_users(self):
        """Return the users that need a recommendation at the current timestep.

        Returns
        -------
        users : iterable
            The users that are online.

        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def users(self):
        """Return all users currently in the environment.

        Returns
        -------
        users : iterable
            All users in the environment.

        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def items(self):
        """Return all items currently in the environment.

        Returns
        -------
        items : iterable
            All items in the environment.

        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ratings(self):
        """Return all ratings that have been made in the environment.

        Returns
        -------
        ratings : iterable
            All ratings in the environment.

        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self):
        """Name of environment, used for saving."""
        raise NotImplementedError

    def seed(self, seed=None):
        """Set the seed the seed for this environment's random number generator(s)."""

    def close(self):
        """Perform any necessary cleanup."""

    def __exit__(self, *args):
        """Perform any necessary cleanup when the object goes out of context."""
        self.close()
        return False


class DictEnvironment(Environment):
    """An environment where data gets passed around as dictionaries.

    Environments can subclass this class by implementing name, true_ratings, _rate_item and
    _reset_state. Optionally environments can also implement _update_state, _rating_env, and
    _select_online_users.

    Parameters
    ----------
    rating_frequency : float
        The proportion of users that will need a recommendation at each step.
        Must be between 0 and 1.
    num_init_ratings : int
        The number of ratings available from the start. User-item pairs are randomly selected.
    memory : int
        The number of recent items a user remembers which affect the rating

    """

    def __init__(self, rating_frequency=0.02, num_init_ratings=0, memory_length=0):
        """Create a Topics environment."""
        self._timestep = -1
        self._random = np.random.RandomState()
        self._rating_frequency = rating_frequency
        self._num_init_ratings = num_init_ratings
        self._users = None
        self._items = None
        self._ratings = None
        self._dense_ratings = None
        self._online_users = None
        self._user_histories = collections.defaultdict(list)
        self._memory_length = memory_length

    def reset(self):
        """Reset the environment to its original state. Must be called before the first step.

        Returns
        -------
        users : OrderedDict
            The initial users where the key represents the user id and the value represents
            the visible features associated with the user.
        items : OrderedDict
            The initial items where the key represents the item id and the value represents
            the visible features associated with the item.
        ratings : dict
            The initial ratings where the key is a double whose first element is the user id
            and the second element is the item id. The value represents the features associated
            with the setting in which the rating was made.

        """
        # Initialize the state of the environment.
        self._timestep = -1
        self._reset_state()
        self._user_histories = collections.defaultdict(list)
        num_users = len(self._users)
        num_items = len(self._items)

        # We will lazily compute dense ratings.
        self._dense_ratings = None

        # Fill the rating dict with initial data.
        idx_1d = self._random.choice(num_users * num_items, self._num_init_ratings,
                                     replace=False)
        user_ids = idx_1d // num_items
        item_ids = idx_1d % num_items
        self._ratings = {}
        for user_id, item_id in zip(user_ids, item_ids):
            self._ratings[user_id, item_id] = (self._rate_item(user_id, item_id),
                                               self._rating_context(user_id))

        # Finally, set the users that will be online for the first step.
        self._online_users = self._select_online_users()

        self._timestep += 1
        return self._users.copy(), self._items.copy(), self._ratings.copy()

    def step(self, recommendations):
        """Run one timestep of the environment.

        Parameters
        ----------
        recommendations : np.ndarray
            The recommendations made to each user. recommendations[i] corresponds to the
            item id recommended to the i-th online user. This array must have the same size as
            the ordered dict returned by online_users.

        Returns
        -------
        users : OrderedDict
            The new users where the key represents the user id and the value represents
            the visible features associated with the user.
        items : OrderedDict
            The new items where the key represents the item id and the value represents
            the visible features associated with the item.
        ratings : dict
            The new ratings where the key is a double whose first element is the user id
            and the second element is the item id. The value represents the features associated
            with the setting in which the rating was made.
        info : dict
            Extra information for debugging and evaluation. info["users"] will return the dict
            of visible user states, info["items"] will return the dict of visible item states, and
            info["ratings"] gets the dict of all ratings.

        """
        assert len(recommendations) == len(self._online_users)
        new_users, new_items = self._update_state()
        # Old dense ratings are now invalid so set it to None and lazily recompute.
        self._dense_ratings = None

        # Get online users to rate the recommended items.
        ratings = {}
        for user_id, item_id in zip(self._online_users, recommendations):
            ratings[user_id, item_id] = (self._rate_item(user_id, item_id),
                                         self._rating_context(user_id))
            self._user_histories[user_id].append(item_id)
            if len(self._user_histories[user_id]) == self._memory_length + 1:
                self._user_histories[user_id].pop(0)
            assert len(self._user_histories[user_id]) <= self._memory_length

        self._ratings.update(ratings)

        # Update the online users.
        self._online_users = self._select_online_users()

        # Create the info dict.
        info = {'users': self._users,
                'items': self._items,
                'ratings': self._ratings}

        self._timestep += 1
        return new_users, new_items, ratings, info

    def online_users(self):
        """Return the users that need a recommendation at the current timestep.

        Returns
        -------
        users_contexts : OrderedDict
            The users that are online. The key is the user id and the value is the
            features that represent the context in which the rating will be made.

        """
        user_contexts = collections.OrderedDict()
        for user_id in self._online_users:
            user_contexts[user_id] = self._rating_context(user_id)
        return user_contexts

    @property
    def users(self):
        """Return all users currently in the environment.

        Returns
        -------
        users : OrderedDict
            All users in the environment, the key represents the user id and the value is the
            visible features associated with the user.

        """
        return self._users

    @property
    def items(self):
        """Return all items currently in the environment.

        Returns
        -------
        items : OrderedDict
            All items in the environment, the key represents the item id and the value is the
            visible features associated with the item.

        """
        return self._items

    @property
    def ratings(self):
        """Return all ratings that have been made in the environment.

        Returns
        -------
        ratings : dict
            All ratings where the key is a double whose first element is the user id
            and the second element is the item id. The value represents the features associated
            with the setting in which the rating was made.

        """
        return self._ratings

    @property
    def dense_ratings(self):
        """Return all the true ratings on every user-item pair at the current timestep.

        A true rating is defined as the rating a user would make with all noise removed.

        Returns
        -------
        dense_ratings : np.ndarray
            The array of all true ratings where true_ratings[i, j] is the rating by user i
            on item j.

        """
        if self._dense_ratings is None:
            self._dense_ratings = self._get_dense_ratings()
        return self._dense_ratings

    def seed(self, seed=None):
        """Set the seed for this environment's random number generator."""
        self._random.seed(seed)

    @abc.abstractmethod
    def _get_dense_ratings(self):
        """Compute all the true ratings on every user-item pair at the current timestep.

        A true rating is defined as the rating a user would make with all noise removed.

        Returns
        -------
        dense_ratings : np.ndarray
            The array of all true ratings where true_ratings[i, j] is the rating by user i
            on item j.

        """
        raise NotImplementedError

    @abc.abstractmethod
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
        rating : float
            The rating the item was given by the user.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _reset_state(self):
        """Reset the state associated with users and items."""
        raise NotImplementedError

    def _update_state(self):  # pylint: disable=no-self-use
        """Update the state associated with users and items.

        The default implementation assumes there is no state that ever gets updated and no new
        users and items ever get added to the environment after being reset. If this is untrue
        you must override this function.

        Returns
        -------
        new_users : OrderedDict
            The newly added users. The key represents the user id and the value
            represents the visible features of the user.
        new_items : OrderedDict
            The newly added items. The key represents the user id and the value
            represents the visible features of the user.

        """
        return collections.OrderedDict(), collections.OrderedDict()

    def _rating_context(self, user_id):  # pylint: disable=no-self-use, unused-argument
        """Get the visible features of the context that the user will make the rating in.

        The default implementation assumes there are no visible features associated with each
        rating. If this is untrue you must override this function.

        Parameters
        ----------
        user_id : int
            The id of the user that will be rating an item.

        Returns
        -------
        context : np.ndarray
            The vector that represents the visible features of the context in which the given user
            will consume and rate the content.

        """
        return np.zeros(0)

    def _select_online_users(self):
        """Select the online users at this timestep.

        Returns
        -------
        online_users : np.ndarray
            The ids of all users that are online.

        """
        num_users = len(self._users)
        num_online = int(self._rating_frequency * num_users)
        return np.random.choice(num_users, size=num_online, replace=False)
