"""Defines a set of base classes from which environments can inherit.

Environment is the interface all environments must implement. The other classes represent
specific environment variants that occur often enough to be abstract classes to inherit from.
"""
import abc


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

    @abc.abstractmethod
    def all_users(self):
        """Return all users currently in the environment.

        Returns
        -------
        users : iterable
            All users in the environment.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def all_items(self):
        """Return all items currently in the environment.

        Returns
        -------
        items : iterable
            All items in the environment.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def all_ratings(self):
        """Return all ratings that have been made in the environment.

        Returns
        -------
        ratings : iterable
            All ratings in the environment.

        """
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
    """An environment where items have a single topic and users prefer certain topics.

    The user preference for any given topic is initialized as Unif(0.5, 5.5) while
    topics are uniformly assigned to items. Users will rate items as clip(p + e, 0, 5)
    where p is their preference for a given topic and e ~ N(0, self._noise).

    Parameters
    ----------
    num_topics : int
        The number of topics items can be assigned to.
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

    """

    def __init__(self, rating_frequency=0.02, num_init_ratings=0):
        """Create a Topics environment."""
        self._timestep = 0
        self._random = np.random.RandomState()
        self._rating_frequency = rating_frequency
        self._num_init_ratings = num_init_ratings
        self._users = None
        self._items = None
        self._ratings = None
        self._online_users = None

    def reset(self):
        """Reset the environment to its original state. Must be called before the first step.

        Returns
        -------
        users : np.ndarray
            This will always be an array where every row has
            size 0 since users don't have features.
        items : np.ndarray
            This will always be an array where every row has
            size 0 since items don't have features.
        ratings : np.ndarray
            The initial ratings where ratings[i, 0] corresponds to the id of the user that
            made the rating, ratings[i, 1] corresponds to the id of the item that was rated
            and ratings[i, 2] is the rating given to that item.

        """
        # Initialize the state of the environment.
        self._timestep = 0
        self._reset_state()
        num_users = len(self._users)
        num_items = len(self._items)

        # Fill the rating dict with initial data.
        idx_1d = np.random.choice(num_users * num_items, self._num_init_ratings,
                                  replace=False)
        user_ids = idx_1d // num_items
        item_ids = idx_1d % num_items
        init_ratings = {}
        for user_id, item_id in zip(user_ids, item_ids):
            ratings[(user_id, item_id)] = self._rate_item(user_id, item_id)

        # Finally, set the users that will be online for the first step.
        num_online = int(self._rating_frequency * num_users)
        self._online_users = np.random.choice(num_users, size=num_online, replace=False)

        return self._users, self._items, init_ratings

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
        users : np.ndarray
            This will always be a size 0 array since no users are ever added.
        items : np.ndarray
            This will always be a size 0 array since no items are ever added.
        ratings : np.ndarray
            New ratings and ratings whose information got updated this timestep. ratings[i, 0]
            is the user id of the i-th online user, ratings[i, 1] is the item they were recommended
            and ratings[i, 2] is the rating they gave the item.
        info : dict
            Extra information for debugging and evaluation. info["users"] will return the array
            of hidden user states, info["items"] will return the array of hidden item states, and
            info["ratings"] gets the sparse matrix of all ratings.

        """
        assert len(recommendations) == len(self._online_users)
        self._timestep += 1
        new_users, new_items = self._update_state()
        # Get online users to rate the recommended items.
        ratings = {}
        for user_id, item_id in zip(self._online_users, recommendations):
            ratings[(user_id, item_id)] = self._rate_item(user_id, item_id)

        # Update the online users.
        num_users = len(self._users)
        num_online = int(self._rating_frequency * num_users)
        self._online_users = np.random.choice(num_users, size=num_online, replace=False)

        # Create the info dict.
        info = {"users": self._users,
                "items": self._items,
                "ratings": self._ratings}

        return new_users, new_items, ratings, info

    def online_users(self):
        """Return the users that need a recommendation at the current timestep.

        Returns
        -------
        users_env : ordered dict
            The users that are online. The key is the user id and the value is the
            features that represent the environment in which the rating will be made.

        """
        user_env = collections.OrderedDict()
        for user_id in self._online_users:
            user_env[user_id] = self._rating_env(user_id)
        return user_env

    def all_users(self):
        """Return all users currently in the environment.

        Returns
        -------
        users : np.ndarray
            All users in the environment, since users don't have features this is just an
            array where each row has size 0.

        """
        return self._users.copy()

    def all_items(self):
        """Return all items currently in the environment.

        Returns
        -------
        items : np.ndarray
            All items in the environment, since items don't have features this is just an
            array where each row has size 0.

        """
        return self._items.copy()

    def all_ratings(self):
        """Return all ratings that have been made in the environment.

        Returns
        -------
        ratings : np.ndarray
            An array where ratings[i, 0] corresponds to the user that made rating i,
            ratings[i, 1] corresponds to the item they rated, and ratings[i, 2] corresponds
            to the rating they gave on a scale of 1-5.

        """
        return self._ratings.copy()

    def seed(self, seed=None):
        """Set the seed for this environment's random number generator."""
        self._random.seed(seed)

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
        rating : int
            The rating the item was given by the user.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _reset_state():
        raise NotImplementedError

    def _update_state():
        return {}, {}

    def _rating_env(user_id):
        return np.zeros(0)
