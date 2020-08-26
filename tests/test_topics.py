"""Tests for the Topics environment."""
import numpy as np

from reclab.environments import Topics
from . import utils

def test_topics_static_simple():
    """Test Topics with only one user."""
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
    assert env.name == 'topics'
    users, items, ratings = env.reset()

    # Test that the users and items have empty features.
    assert users[0].shape == (0,)
    assert items[0].shape == (0,)
    assert env.online_users[0].shape == (0,)

    # Test that item topics and user preferences are of the correct size
    assert env._item_topics.shape == (2,)
    assert env._user_preferences.shape == (1,2)
    old_user_preferences = env._user_preferences

    old_dense_ratings = env._get_dense_ratings()

    # Recommend item 0, we shouldn't observe new users or items.
    users, items, ratings, _ = env.step(np.array([[0]]))
    assert users == {}
    assert items == {}

    # Test that the preferences didn't change
    assert np.array_equal(old_user_preferences, env._user_preferences)

    # Test that old dense ratings are not changing
    assert np.array_equal(old_dense_ratings, env._get_dense_ratings())


test_topics_static_simple()
