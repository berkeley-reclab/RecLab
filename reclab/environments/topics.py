"""Contains the implementation for the Topics environment.

In this environment users have a hidden preference for each topic and each item has a
hidden topic assigned to it.
"""
import collections
import numpy as np

from . import environment


class Topics(environment.DictEnvironment):
    """
    An environment where items have a single topic and users prefer certain topics.

    The user preference for any given topic is initialized as Unif(0.5, 5.5) while
    topics are uniformly assigned to items. Users will
    also have a changing preference for topics they get recommended based on the topic_change
    parameter. Users and items can have biases, there can also exist an underlying bias.

    Ratings are generated as
    r = clip( user preference for a given topic + b_u + b_i + b_0, 1, 5)
    where b_u is a user bias, b_i is an item bias, and b_0 is a global bias.

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
    user_dist_choice : str
        The choice of user distribution for selecting online users. By default, the subset of
        online users is chosen from a uniform distribution. Currently supports normal and lognormal.
    shift_steps : int
        The number of timesteps to wait between each user preference shift.
    shift_frequency : float
        The proportion of users whose preference we wish to change during a preference shift.
    shift_weight : float
        The weight to assign to a user's new preferences after a preference shift.
        User's old preferences get assigned a weight of 1 - shift_weight.
    user_bias_type : normal or power
        distribution type for user biases.
        normal is normal distribution with default mean zero and variance 0.5
        power is power law distribution
    item_bias_type : normal or power
        distribution type for item biases.
        normal is normal distribution with default mean zero and variance 0.5
        power is power law distribution

    """

    def __init__(self,
                 num_topics,
                 num_users,
                 num_items,
                 rating_frequency=1.0,
                 num_init_ratings=0,
                 noise=0.0,
                 topic_change=0.0,
                 memory_length=0,
                 boredom_threshold=0,
                 boredom_penalty=0.0,
                 user_dist_choice='uniform',
                 shift_steps=1,
                 shift_frequency=0.0,
                 shift_weight=0.0,
                 user_bias_type='normal',
                 item_bias_type='normal'):
        """Create a Topics environment."""
        super().__init__(rating_frequency, num_init_ratings, memory_length, user_dist_choice)
        self._num_topics = num_topics
        self._num_users = num_users
        self._num_items = num_items
        self._topic_change = topic_change
        self._noise = noise
        self._user_preferences = None
        self._item_topics = None
        self._boredom_threshold = boredom_threshold
        self._boredom_penalty = boredom_penalty
        self._shift_steps = shift_steps
        self._shift_frequency = shift_frequency
        self._shift_weight = shift_weight
        self._user_biases = None
        self._item_biases = None
        self._offset = None
        self._user_bias_type = user_bias_type
        self._item_bias_type = item_bias_type

    @property
    def name(self):  # noqa: D102
        return 'topics'

    def _get_dense_ratings(self):  # noqa: D102
        ratings = np.zeros([self._num_users, self._num_items])
        for item_id in range(self._num_items):
            topic = self._item_topics[item_id]
            ratings[:, item_id] = (self._user_preferences[:, topic] +
                                   np.full((self._num_users), self._item_biases[item_id]) +
                                   self._user_biases + np.full((self._num_users), self._offset))

        # Account for boredom.
        for user_id in range(self._num_users):
            recent_topics = [self._item_topics[item] for item in self._user_histories[user_id]]
            recent_topics, counts = np.unique(recent_topics, return_counts=True)
            recent_topics = recent_topics[counts > self._boredom_threshold]
            for topic_id in recent_topics:
                ratings[user_id, self._item_topics == topic_id] -= self._boredom_penalty

        return ratings

    def _get_rating(self, user_id, item_id):  # noqa: D102
        topic = self._item_topics[item_id]
        rating = (self._user_preferences[user_id, topic] + self._user_biases[user_id] +
                  self._item_biases[item_id] + self._offset)
        recent_topics = [self._item_topics[item] for item in self._user_histories[user_id]]
        if recent_topics.count(topic) > self._boredom_threshold:
            rating -= self._boredom_penalty
        rating = np.clip(rating + self._dynamics_random.randn() * self._noise, 1, 5)
        return rating

    def _rate_items(self, user_id, item_ids):  # noqa: D102
        # TODO: Add support for slates of size greater than 1.
        item_id = [item_ids[0]]
        rating = self._get_rating(user_id, item_id)
        # Updating underlying preference
        topic = self._item_topics[item_id]
        preference = self._user_preferences[user_id, topic]
        if preference <= 5:
            self._user_preferences[user_id, topic] += self._topic_change
            not_topic = np.arange(self._num_topics) != topic
            self._user_preferences[user_id, not_topic] -= (
                self._topic_change / (self._num_topics - 1))
        return rating

    def _reset_state(self):  # noqa: D102
        if self._user_bias_type == 'normal':
            self._user_biases = self._init_random.normal(loc=0., scale=0.5, size=self._num_users)
        elif self._user_bias_type == 'power':
            self._user_biases = 1-self._init_random.power(5, size=self._num_users)
        else:
            print('User bias distribution is not supported')

        if self._item_bias_type == 'normal':
            self._item_biases = self._init_random.normal(loc=0., scale=0.5, size=self._num_items)
        elif self._item_bias_type == 'power':
            self._item_biases = 1-self._init_random.power(5, size=self._num_users)
        else:
            print('Item bias distribution is not supported')

        self._offset = 0
        self._user_preferences = self._init_random.uniform(low=0.5, high=5.5,
                                                           size=(self._num_users, self._num_topics))
        self._item_topics = self._init_random.choice(self._num_topics, size=self._num_items)
        self._users = collections.OrderedDict((user_id, np.zeros(0))
                                              for user_id in range(self._num_users))
        self._items = collections.OrderedDict((item_id, np.zeros(0))
                                              for item_id in range(self._num_items))

    def _update_state(self):  # noqa: D102
        if (self._timestep + 1) % self._shift_steps == 0:
            # Apply preference and bias shift to a fraction of users.

            shifted_users = self._dynamics_random.choice(
                self._num_users, int(self._num_users * self._shift_frequency))

            new_preferences = self._init_random.uniform(low=0.5, high=5.5,
                                                        size=(len(shifted_users), self._num_topics))
            if self._user_bias_type == 'normal':
                new_user_biases = self._init_random.normal(loc=0, scale=0.5,
                                                           size=len(shifted_users))
            elif self._user_bias_type == 'power':
                new_user_biases = 1-self._init_random.power(5, size=len(shifted_users))
            else:
                print('User bias distribution is not supported')

            self._user_preferences[shifted_users] = (
                self._shift_weight * self._user_preferences[shifted_users] +
                (1 - self._shift_weight) * new_preferences)

            self._user_biases[shifted_users] = (
                self._shift_weight * self._user_biases[shifted_users] +
                (1 - self._shift_weight) * new_user_biases)

        return collections.OrderedDict(), collections.OrderedDict()
