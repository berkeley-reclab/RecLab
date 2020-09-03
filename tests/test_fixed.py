"""Tests for the FixedRating environment.

The primary intent of these tests is to validate the code in the DictEnvironment
parent class instead of only testing FixedRating.
"""
import numpy as np

from reclab.environments import FixedRating
from . import utils


def test_fixed_simple():
    """Test FixedRating with only two items."""
    env = FixedRating(num_users=1,
                      num_items=2,
                      rating_frequency=1.0,
                      num_init_ratings=0)
    assert env.name == 'fixed'
    users, items, ratings = env.reset()

    # Test that the users and items have empty features.
    assert users[0].shape == (0,)
    assert items[0].shape == (0,)
    assert env.online_users[0].shape == (0,)

    # Recommend item 0, we shouldn't observe new users or items.
    users, items, ratings, _ = env.step(np.array([[0]]))
    assert users == {}
    assert items == {}

    # Test that item 0 will have a rating of 1.
    assert ratings[(0, 0)][0] == 1

    # Recommend item 1, the environment should rate it 5.
    users, items, ratings, _ = env.step(np.array([[1]]))
    assert users == {}
    assert items == {}
    assert ratings[(0, 1)][0] == 5

    # Test the internal state of the environment.
    assert len(env.users) == 1
    assert env.users[0].shape == (0,)
    assert len(env.items) == 2
    assert env.items[0].shape == (0,)
    assert len(env.ratings) == 2
    assert env.ratings[0, 0][0] == 1
    assert env.ratings[0, 1][0] == 5


def test_fixed_two_users(mocker):
    """Test FixedRating with two users."""
    mocker.patch('reclab.environments.FixedRating._select_online_users',
                 utils.mock_select_online_users)
    env = FixedRating(num_users=2,
                      num_items=2,
                      rating_frequency=0.5,
                      num_init_ratings=0)
    env.reset()
    assert env.dense_ratings.shape == (2, 2)
    assert (env.dense_ratings[:, 0] == 1).all()
    assert (env.dense_ratings[:, 1] == 5).all()
    assert len(env.online_users) == 1
    assert 0 in env.online_users
    env.step(np.array([[0]]))
    assert len(env.online_users) == 1
    assert 1 in env.online_users
    env.step(np.array([[1]]))
    assert len(env.ratings) == 2
    assert env.ratings[0, 0][0] == 1
    assert env.ratings[1, 1][0] == 5


def test_fixed_slates():
    """Test FixedRating with slate recommendations."""
    env = FixedRating(num_users=1,
                      num_items=4,
                      rating_frequency=1.0,
                      num_init_ratings=0)
    env.reset()
    _, _, ratings, _ = env.step(np.array([[0, 1, 2, 3]]))
    assert len(ratings) == 1
    assert ratings[0, 3][0] == 5
    _, _, ratings, _ = env.step(np.array([[0, 1, 2, 3]]))
    assert len(ratings) == 1
    assert ratings[0, 2][0] == 5
    _, _, ratings, _ = env.step(np.array([[0, 2, 3]]))
    assert len(ratings) == 1
    assert ratings[0, 0][0] == 1
    _, _, ratings, _ = env.step(np.array([[0, 1, 2, 3]]))
    assert len(ratings) == 1
    assert ratings[0, 1][0] == 1


def test_init_ratings():
    """Test FixedRating properly initializes ratings."""
    env = FixedRating(num_users=50,
                      num_items=50,
                      rating_frequency=1.0,
                      num_init_ratings=100)
    env.seed(0)
    _, _, ratings = env.reset()
    assert len(ratings) == 100
    for (user_id, item_id), (rating, context) in ratings.items():
        assert context.shape == (0,)
        assert user_id < 50
        assert item_id < 50
        if rating == 5.0:
            assert item_id >= 25
        else:
            assert item_id < 25
