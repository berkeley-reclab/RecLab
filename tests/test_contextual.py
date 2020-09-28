"""Tests for the Contextual environment."""
import numpy as np

from reclab.environments import Contextual


def test_contextual_wiki():
    """Test contextual instantiated with Wiki10-31k."""
    env = Contextual('wiki10-31k')
    assert env.name == 'contextual'
    users, items, ratings = env.reset()

    # Test that the users and items have empty features.
    assert users[0].shape == (0,)
    assert items[0].shape == (0,)

    # Test that contexts have a given size.
    assert env.online_users[0].shape == (101938,)
    context = env.online_users[0]

    # Test the number of users and items.
    assert len(env.online_users) == 1
    assert len(users) == 1
    assert len(items) == 30938

    # Recommend item 0, we should a new user and no new items.
    users, items, ratings, _ = env.step(np.array([[0]]))
    assert len(users) == 1
    assert 1 in users
    assert len(items) == 0

    # The first user should have left.
    assert 0 not in env.users

    # We should only have received one rating of 0.
    assert len(ratings) == 1
    assert ratings[(0, 0)][0] == 0.0
    assert np.array_equal(ratings[(0, 0)][1], context)
