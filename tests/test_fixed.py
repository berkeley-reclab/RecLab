"""Tests for the FixedRating environment."""
import numpy as np

from reclab.environments import FixedRating


def test_fixed_simple():
    """Test FixedRating with only two items."""
    env = FixedRating(num_users=1,
                      num_items=2,
                      rating_frequency=1.0,
                      num_init_ratings=0)
    assert env.name == 'fixed'
    users, items, ratings = env.reset()
    assert users[0].shape == (0,)
    assert items[0].shape == (0,)
    assert env.online_users()[0].shape == (0,)
    users, items, ratings, info = env.step(np.array([[0]]))
    assert users == {}
    assert items == {}
    assert ratings[(0, 0)][0] == 1
    users, items, ratings, info = env.step(np.array([[1]]))
    assert users == {}
    assert items == {}
    assert ratings[(0, 1)][0] == 5
