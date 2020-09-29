"""Defines a set of base classes from which environments can inherit.

Environment is the interface all environments must implement. The other classes represent
specific environment variants that occur often enough to be abstract classes to inherit from.
"""
import abc
import collections

import numpy as np
import scipy.stats


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

    @property
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
    user_dist_choice : str
        The choice of user distribution for selecting online users. By default, the subset of
        online users is chosen from a uniform distribution. Currently supports normal and lognormal.

    """

    def __init__(self, rating_frequency=0.02, num_init_ratings=0, memory_length=0,
                 user_dist_choice='uniform'):
        """Create a new DictEnvironment."""
        self._timestep = -1
        # The RandomState to use while initializing the environment.
        self._init_random = np.random.RandomState()
        # The RandomState to use after the environment is initialized.
        self._dynamics_random = np.random.RandomState()
        self._rating_frequency = rating_frequency
        self._num_init_ratings = num_init_ratings
        self._user_dist_choice = user_dist_choice
        self._users = None
        self._items = None
        self._ratings = None
        self._rated_items = None
        self._dense_ratings = None
        self._online_users = None
        self._user_prob = None
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
        self._user_prob = self._get_user_prob()

        # We will lazily compute dense ratings.
        self._dense_ratings = None

        # Fill the rating dict with initial data.
        idx_1d = self._init_random.choice(num_users * num_items, self._num_init_ratings,
                                          replace=False,
                                          p=np.repeat(self._user_prob, num_items) / num_items)
        user_ids = idx_1d // num_items
        item_ids = idx_1d % num_items
        self._ratings = {}

        for user_id, item_id in zip(user_ids, item_ids):
            # TODO: This is a hack, but I don't think we should necessarily put the burden
            # of having to implement a version of _rate_item that knows whether it's being called
            # in reset or not on people deriving from this class. Need to think of a better way
            # than doing this though.
            temp_random = self._dynamics_random
            self._dynamics_random = self._init_random
            self._ratings[user_id, item_id] = (self._rate_items(user_id, np.array([item_id]))[0],
                                               self._rating_context(user_id))
            self._dynamics_random = temp_random

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
            item ids recommended to the i-th online user. The first dimension of this array
            must have the same size as the ordered dict returned by online_users.

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
        # Old dense ratings are now invalid so set it to None and lazily recompute.
        self._dense_ratings = None

        # Get online users to rate the recommended items.
        ratings = {}
        for user_id, item_ids in zip(self._online_users, recommendations):
            user_context = self._rating_context(user_id)
            user_ratings = self._rate_items(user_id, item_ids)
            for item_id, rating in zip(item_ids, user_ratings):
                # If a rating is NaN the user did not rate the item.
                if not np.isnan(rating):
                    ratings[user_id, item_id] = (rating, user_context)
            self._user_histories[user_id].append(item_ids)
            if len(self._user_histories[user_id]) == self._memory_length + 1:
                self._user_histories[user_id].pop(0)
            assert len(self._user_histories[user_id]) <= self._memory_length

        self._ratings.update(ratings)

        # Create the info dict.
        info = {'users': self._users,
                'items': self._items,
                'ratings': self._ratings}

        # Update the user and item state.
        new_users, new_items = self._update_state()

        # Update the online users.
        self._online_users = self._select_online_users()

        self._timestep += 1

        return new_users, new_items, ratings, info

    @property
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
        """Set the seed for this environment's random number generator.

        Parameters
        ----------
        seed : int or tuple of int
            The seed for the random number generators. If seed is an int or a tuple of length 1 all
            random number generators will be initialized with that seed. If it is a tuple of length
            2 the random number generator for the initial state of the environment will be
            initialized with seed[0] and the random number generator for the environment dynamics
            will be initialized with seed[1].

        """
        if seed is None or np.issubdtype(type(seed), np.integer):
            self._init_random.seed(seed)
            self._dynamics_random.seed(seed)
        elif len(seed) == 1:
            self._init_random.seed(seed[0])
            self._dynamics_random.seed(seed[0])
        else:
            self._init_random.seed(seed[0])
            self._dynamics_random.seed(seed[1])

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
    def _rate_items(self, user_id, item_ids):
        """Get a user to rate an item and update the internal rating state.

        Parameters
        ----------
        user_id : int
            The id of the user making the rating.
        item_ids : iterable of int
            The ids of the items being rated.

        Returns
        -------
        rating : iterable float
            The ratings the items were given by the user. Can include np.nan entries
            if the user did not rate a given item.

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

    def _get_user_prob(self):
        """Get the probability distribution for choosing online users at each timestep.

        The default assumes that users are drawn at uniform. To modify, change the parameters
        _user_dist when initializing the environment.

        """
        dist_choice = self._user_dist_choice
        num_users = len(self._users)

        if dist_choice == 'uniform':
            user_dist = np.ones(num_users) / num_users
        elif dist_choice == 'norm':
            idx = np.random.permutation(num_users)
            user_dist = np.array([
                scipy.stats.norm.pdf(idx[i], scale=num_users / 7, loc=num_users / 2)
                for i in range(num_users)])
            user_dist = user_dist / sum(user_dist)
            user_dist = np.clip(user_dist, 0, 1)
        elif dist_choice == 'lognorm':
            idx = np.random.permutation(num_users)
            user_dist = np.array([scipy.stats.lognorm.pdf(idx[i], 1, scale=num_users / 7, loc=-1)
                                  for i in range(num_users)])
            user_dist = user_dist / sum(user_dist)
            user_dist = np.clip(user_dist, 0, 1)
        elif dist_choice == 'pareto':
            idx = np.random.permutation(num_users)
            user_dist = np.array([scipy.stats.pareto.pdf(idx[i], 1, scale=num_users / 1e4, loc=-1)
                                  for i in range(num_users)])
            user_dist = user_dist / sum(user_dist)
            user_dist = np.clip(user_dist, 0, 1)
        else:
            raise ValueError('user distribution name not recognized')

        return user_dist

    def _select_online_users(self):
        """Select the online users at this timestep.

        Returns
        -------
        online_users : np.ndarray
            The ids of all users that are online.

        """
        user_ids = list(self._users.keys())
        num_online = int(self._rating_frequency * len(self._users))
        return self._dynamics_random.choice(user_ids, size=num_online,
                                            replace=False, p=self._user_prob)
