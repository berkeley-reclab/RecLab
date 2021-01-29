"""Tests for the BetaRank environment."""
import numpy as np

from reclab.environments import BetaRank


def test_beta_simple():
    """Test BetaRank with only one user."""
    env = BetaRank(dimension=10,
                   num_users=1,
                   num_items=2,
                   rating_frequency=1.0,
                   num_init_ratings=0)
    assert env.name == 'beta-rank'
    users, items, ratings = env.reset()

    # Test that the users and items have empty features.
    assert users[0].shape == (0,)
    assert items[0].shape == (0,)
    assert env.online_users[0].shape == (0,)

    # Recommend item 0, we shouldn't observe new users or items.
    users, items, ratings, _ = env.step(np.array([[0]]))
    assert users == {}
    assert items == {}

    # Test that item 0 falls in the [0, 1] range.
    assert ratings[(0, 0)][0] <= 1 and ratings[(0, 0)][0] >= 0


def test_fixed_slates():
    """Test FixedRating with slate recommendations."""
    env = BetaRank(dimension=10,
                   num_users=1,
                   num_items=100,
                   rating_frequency=1.0,
                   num_init_ratings=0)
    env.seed(0)
    env.reset()
    assert ((env.dense_ratings >= 0) & (env.dense_ratings <= 1)).all()
    # Sort item ids from best to worst.
    item_ids = env.dense_ratings[0].argsort()
    # Swap the second largest and second smallest elements.
    item_ids[1], item_ids[-2] = item_ids[-2], item_ids[1]
    # The environment should pick the second item here since it will
    # have a larger value than other highly ranked items.
    _, _, ratings, _ = env.step(np.array([item_ids]))
    assert len(ratings) == 1
    assert (0, item_ids[1]) in ratings
    # Swap the tenth largest and tenth smallest elements.
    item_ids[9], item_ids[-10] = item_ids[-10], item_ids[9]
    # The environment should pick the tenth item here since it will
    # have a larger value than other highly ranked items, except for
    # the second item which has already been rated.
    _, _, ratings, _ = env.step(np.array([item_ids]))
    assert len(ratings) == 1
    assert (0, item_ids[9]) in ratings
