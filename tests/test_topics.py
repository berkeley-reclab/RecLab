# pylint: disable=protected-access
"""Tests for the Topics environment."""
import copy
import numpy as np

from reclab.environments import Topics


def _test_dimension_consistency(environment):
    """ Basic Helper Test to check if dimension of
    various environment properties."""
    env = copy.deepcopy(environment)

    assert env.name == 'topics'
    users, items, _ = env.reset()

    # Test that the users and items have empty features.
    num_users = len(env.users)
    num_items = len(env.items)
    num_topics = env._num_topics
    assert users[0].shape == (0,)
    assert items[0].shape == (0,)
    assert env.online_users[0].shape == (0,)

    # Test that item topics and user preferences are of the correct size.
    assert env._item_topics.shape == (num_items,)
    assert env._user_preferences.shape == (num_users, num_topics)

    # Recommend item 0, we shouldn't observe new users or items.
    users, items, _, _ = env.step(np.array([[0]]))
    assert users == {}
    assert items == {}


def test_topics_static_simple():
    """Test Topics with only one user, with no preference shifts
    and no topic change and no boredom."""
    env = Topics(num_topics=2,
                 num_users=1,
                 num_items=2,
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
                 shift_weight=0.0)

    _test_dimension_consistency(env)
    env.reset()

    old_user_preferences = copy.deepcopy(env._user_preferences)
    old_dense_ratings = env._get_dense_ratings()

    # Recommend item 0
    env.step(np.array([[0]]))

    # Test that the preferences didn't change
    assert np.array_equal(old_user_preferences, env._user_preferences)
    # Test that the dense ratings didn't change
    assert np.array_equal(old_dense_ratings, env._get_dense_ratings())


def test_topics_shift():
    """Test Topics with random preference shifts"""
    env = Topics(num_topics=2,
                 num_users=1,
                 num_items=10,
                 rating_frequency=1.0,
                 num_init_ratings=0,
                 noise=0.0,
                 topic_change=0.0,
                 memory_length=0,
                 boredom_threshold=0,
                 boredom_penalty=0.0,
                 user_dist_choice='uniform',
                 shift_steps=2,
                 shift_frequency=1,
                 shift_weight=0.5)

    _test_dimension_consistency(env)
    env.reset()

    old_user_preferences = copy.deepcopy(env._user_preferences)
    old_user_biases = copy.deepcopy(env._user_biases)

    # Recommend item 0.
    env.step(np.array([[0]]))

    # Test that the preferences and biases didn't change.
    assert np.array_equal(old_user_preferences, env._user_preferences)
    assert np.array_equal(old_user_biases, env._user_biases)

    # Recommend another item and check that preferences have changed.
    env.step(np.array([[1]]))
    assert not np.array_equal(old_user_preferences, env._user_preferences)
    assert not np.array_equal(old_user_biases, env._user_biases)


def test_topics_boredom():
    """Test Topics with boredom shifts"""
    env = Topics(num_topics=2,
                 num_users=1,
                 num_items=10,
                 rating_frequency=1.0,
                 num_init_ratings=0,
                 noise=0.0,
                 topic_change=0.0,
                 memory_length=3,
                 boredom_threshold=1,
                 boredom_penalty=1,
                 user_dist_choice='uniform',
                 shift_steps=1,
                 shift_frequency=0,
                 shift_weight=0)

    _test_dimension_consistency(env)
    env.reset()
    # Change all the item types to type 0.
    env._item_topics = np.zeros(len(env.items), dtype=int)

    old_ratings = env._get_dense_ratings()

    # Recommend item 0 and check that ratings don't change.
    env.step(np.array([[0]]))
    assert np.array_equal(old_ratings, env._get_dense_ratings())

    # Recommend item 1 and check that dense ratings decrease by the
    # same amount as the boredom penalty.
    env.step(np.array([[1]]))
    assert np.array_equal(old_ratings-env._boredom_penalty, env._get_dense_ratings())


def test_topics_change():
    """Test Topics with topic change"""
    env = Topics(num_topics=2,
                 num_users=1,
                 num_items=10,
                 rating_frequency=1.0,
                 num_init_ratings=0,
                 noise=0.0,
                 topic_change=0.5,
                 memory_length=0,
                 boredom_threshold=0,
                 boredom_penalty=0,
                 user_dist_choice='uniform',
                 shift_steps=1,
                 shift_frequency=0,
                 shift_weight=0)

    _test_dimension_consistency(env)
    env.reset()
    # Change all the item types to type 0.
    env._item_topics = np.zeros(len(env.items), dtype=int)

    old_user_preferences = copy.deepcopy(env._user_preferences)

    # Recommend item 0 and check that preferences for the recommended topic have
    # increased while the preference for the other topic decreased.
    env.step(np.array([[0]]))
    topic = env._item_topics[0]
    new_user_preferences = env._user_preferences
    assert new_user_preferences[0][topic] >= old_user_preferences[0][topic]
    assert new_user_preferences[0][1-topic] <= old_user_preferences[0][1-topic]
