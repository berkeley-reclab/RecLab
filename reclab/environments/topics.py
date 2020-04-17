"""Contains the implementation for the Topics environment.

In this environment users have a hidden preference for each topic and each item has a
hidden topic assigned to it.
"""
import numpy as np

from . import environment


class Topics(environment.DictEnvironment):
    """An environment where items have a single topic and users prefer certain topics.

    The user preference for any given topic is initialized as Unif(0.5, 5.5) while
    topics are uniformly assigned to items. Users will rate items as clip(p + e, 0, 5)
    where p is their preference for a given topic and e ~ N(0, self._noise). Users will
    also have a changing preference for topics they get recommended based on the topic_change
    parameter.

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
    topic_change : float
        How much the user's preference for a topic changes each time that topic is recommended
        to them. The negative of topic_change gets split across all other topics as well.
    memory_length : int
        The number of recent topics a user remembers which affect the rating
    boredom_threshold : int
        The number of times a topics has to be seen within the memory to gain a
        penalty.
    boredom_penalty : float
        The penalty on the rating when a user is bored

    """

    def __init__(self, num_topics, num_users, num_items, rating_frequency=1.0,
                 num_init_ratings=0, noise=0.0, topic_change=0.0, memory_length=0,
                 boredom_threshold=0, boredom_penalty=0.0):
        """Create a Topics environment."""
        super().__init__(rating_frequency, num_init_ratings, memory_length)
        self._num_topics = num_topics
        self._num_users = num_users
        self._num_items = num_items
        self._topic_change = topic_change
        self._noise = noise
        self._user_preferences = None
        self._item_topics = None
        self._boredom_threshold = boredom_threshold
        self._boredom_penalty = boredom_penalty

    @property
    def name(self):
        """Name of environment, used for saving."""
        return 'topics'

    def dense_ratings(self):  # noqa: D102
        ratings = np.zeros([self._num_users, self._num_items])
        for item_id in range(self._num_items):
            topic = self._item_topics[item_id]
            ratings[:, item_id] = self._user_preferences[:, topic]
        return ratings

    def _rate_item(self, user_id, item_id):
        """Get a user to rate an item and update the internal rating state."""
        topic = self._item_topics[item_id]
        preference = self._user_preferences[user_id, topic]
        rating = np.clip(np.round(preference + self._random.randn() * self._noise), 1, 5)
        recent_topics = [self._item_topics[item] for item in self._user_histories[user_id]]
        if recent_topics.count(topic) > self._boredom_threshold:
            rating -= self._boredom_penalty
        rating = np.clip(rating, 1, 5)
        # Updating underlying preference
        if preference <= 5:
            self._user_preferences[user_id, topic] += self._topic_change
            self._user_preferences[user_id, np.arange(self._num_topics) != topic] -= (
                self._topic_change / (self._num_topics - 1))
        return rating

    def _reset_state(self):
        """Reset the state of the environment."""
        self._user_preferences = np.random.uniform(low=0.5, high=5.5,
                                                   size=(self._num_users, self._num_topics))
        self._item_topics = np.random.choice(self._num_topics, size=self._num_items)
        self._users = {user_id: np.zeros((0,)) for user_id in range(self._num_users)}
        self._items = {item_id: np.zeros((0,)) for item_id in range(self._num_items)}
