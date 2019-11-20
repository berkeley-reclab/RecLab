import numpy as np
import scipy

from reclab.environments.environment import Environment


class RandomPreferences(Environment):
    def __init__(self, num_topics, num_users, num_items,
                 rating_frequency=0.2, num_init_ratings=0):
        self._random = np.random.RandomState()
        self._noise = 1.0
        self._num_topics = num_topics
        self._num_users = num_users = num_users
        self._num_items = num_items
        self._rating_frequency=rating_frequency
        self._num_init_ratings = num_init_ratings
        self._users = None
        self._items = None
        self._ratings = None
        self._timestep = 0

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
        # Users have a 1-5 uniformly distributed preference for each topic.
        self._users = np.random.uniform(low=1, high=5.0, size=(self._num_users, self._num_topics))
        # Randomly sample a single topic for each item.
        self._items = np.random.choice(self._num_topics, size=self._num_items)
        # Create the initially empty matrix of user-item ratings.
        self._ratings = scipy.sparse.dok_matrix((self._num_users, self._num_items))
        self._timestep = 0

        # Fill the rating array with initial data.
        init_ratings = np.zeros((self._num_init_ratings, 3))
        rated_idxs = set()
        for i in range(self._num_init_ratings):
            # Keep sampling rating indices until we've reached an uninitialized one.
            while True:
                user_id = np.random.choice(self._num_users)
                item_id = np.random.choice(self._num_items)
                if (user_id, item_id) not in rated_idxs:
                    break
            rated_idxs.add((user_id, item_id))
            init_ratings[i, 0] = user_id
            init_ratings[i, 1] = item_id
            init_ratings[i, 2] = self._rate_item(user_id, item_id)

        # Finally, set the users that will be online for the first step.
        num_online = int(self._rating_frequency * self._num_users)
        self._online_users = np.random.choice(self._num_users, size=num_online, replace=False)

        users = {}
        for i in range(self._num_users):
            users[i] = np.zeros((0))

        items = {}
        for i in range(self._num_items):
            items[i] = np.zeros((0))

        return users, items, init_ratings

    def step(self, recommendations):
        """Run one timestep of the environment.

        Parameters
        ----------
        recommendations : np.ndarray
            The recommendations made to each user. recommendations[i] corresponds to the
            item id recommended to the i-th online user. This array must have the same size as
            the array returned by online_users.

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
        # Get online users to rate the recommended items.
        assert(len(recommendations) == len(self._online_users))
        ratings = np.zeros((len(recommendations), 3), dtype=np.int)
        ratings[:, 0] = self._online_users 
        ratings[:, 1] = recommendations.squeeze()
        for i in range(len(recommendations)):
            user_id = ratings[i, 0]
            item_id = ratings[i, 1]
            ratings[i, 2] = self._rate_item(user_id, item_id)

        if np.random.randint(low=1, high=100) < 25:
            self._users = np.random.uniform(low=1, high=5.0, size=(self._num_users, self._num_topics))
        # Update the online users.
        num_online = int(self._rating_frequency * self._num_users)
        self._online_users = np.random.choice(self._num_users, size=num_online, replace=False)

        # Create the info dict.
        info = {"users": self._users,
                "items": self._items,
                "ratings": self._ratings}
        self._timestep += 1

        return {}, {}, ratings, info

    def online_users(self):
        """Return the users that need a recommendation at the current timestep.

        Returns
        -------
        users : np.ndarray
            The user ids of the users that are online.
        """
        user_env = {}
        for user_id in self._online_users:
            user_env[user_id] = np.zeros(0)
        return user_env

    def all_users(self):
        """Return all users currently in the environment.

        Returns
        -------
        users : np.ndarray
            All users in the environment, since users don't have features this is just an
            array where each row has size 0.
        """
        return np.zeros((self._num_users, 0))

    def all_items(self):
        """Return all items currently in the environment.

        Returns
        -------
        items : np.ndarray
            All items in the environment, since items don't have features this is just an
            array where each row has size 0.
        """
        return np.zeros((self._num_items, 0))

    def all_ratings(self):
        """Return all ratings that have been made in the environment.

        Returns
        -------
        ratings : np.ndarray
            An array where ratings[i, 0] corresponds to the user that made rating i,
            ratings[i, 1] corresponds to the item they rated, and ratings[i, 2] corresponds
            to the rating they gave on a scale of 1-5.
        """
        ratings = np.zeros((self._ratings.nnz(), 3), dtype=np.int)
        for i, user_id, item_id in enumerate(self._ratings.nonzero()):
            ratings[i, 0] = user_id
            ratings[i, 1] = item_id
            ratings[i, 2] = self._ratings[user_id, item_id]
        return ratings

    def seed(self, seed=None):
        """Set the seed for this environment's random number generator."""
        self._random.seed(seed)

    def redistribute(self, preferences, topic, epsilon=0.1):
        preferences = [i - epsilon/len(preferences) for i in preferences]
        preferences[topic] = max(5.0, preferences[topic] + epsilon)
        return preferences

    def _rate_item(self, user_id, item_id, epsilon=0.05):
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
        topic = self._items[item_id]
        preference = self._users[user_id, topic]
        self._users[user_id] = self.redistribute(self._users[user_id], topic)
        rating = np.clip(preference + self._random.randn() * self._noise, 0, 5).astype(np.int)
        self._ratings[user_id, item_id] = rating
        return rating
