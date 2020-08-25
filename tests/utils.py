"""A set of utility functions for testing."""
import collections
import numpy as np

from reclab import data_utils

NUM_USERS_ML100K = 943
NUM_ITEMS_ML100K = 1682

NUM_USERS_SIMPLE = 2
NUM_ITEMS_SIMPLE = 3


def test_predict_ml100k(recommender, rmse_threshold=1.1, seed=None, test_dense=False):
    """Test that recommender predicts well and that it gets better with more data."""
    users, items, ratings = data_utils.read_dataset('ml-100k')
    assert NUM_USERS_ML100K == len(users)
    assert NUM_ITEMS_ML100K == len(items)
    train_ratings, test_ratings = data_utils.split_ratings(ratings, 0.9, shuffle=True, seed=seed)
    train_ratings_1, train_ratings_2 = data_utils.split_ratings(train_ratings, 0.5)
    recommender.reset(users, items, train_ratings_1)
    user_item = [(key[0], key[1], val[1]) for key, val in test_ratings.items()]
    preds = recommender.predict(user_item)
    targets = [t[0] for t in test_ratings.values()]
    rmse1 = rmse(preds, targets)

    # We should get a relatively low RMSE here.
    assert rmse1 < rmse_threshold

    recommender.update(ratings=train_ratings_2)
    preds = recommender.predict(user_item)
    rmse2 = rmse(preds, targets)

    # The RMSE should have reduced.
    assert rmse1 > rmse2

    if test_dense:
        # Test that the dense predictions work as well.
        dense = recommender.dense_predictions
        preds = np.array([dense[key[0] - 1, key[1] - 1] for key in test_ratings])
        rmse3 = rmse(preds, targets)
        # The RMSE should have reduced.
        assert rmse1 > rmse3


def test_binary_recommend_ml100k(recommender, hit_rate_threshold, seed=None):
    """Test that the recommender will recommend good items and it gets better with more data."""
    users, items, ratings = data_utils.read_dataset('ml-100k')
    assert NUM_USERS_ML100K == len(users)
    assert NUM_ITEMS_ML100K == len(items)
    train_ratings, test_ratings = data_utils.split_ratings(ratings, 0.9, shuffle=True, seed=seed)
    train_ratings_1, train_ratings_2 = data_utils.split_ratings(train_ratings, 0.5)
    all_contexts = collections.OrderedDict([(user_id, np.zeros(0)) for user_id in users])

    recommender.reset(users, items, train_ratings_1)
    recs, _ = recommender.recommend(all_contexts, 1)
    num_hits = sum((user_id, rec) in test_ratings for user_id, rec in zip(users, recs[:, 0]))
    hit_rate1 = num_hits / NUM_USERS_ML100K

    # We should get a relatively low hit rate here.
    assert hit_rate1 > hit_rate_threshold, hit_rate1

    recommender.reset(users, items, train_ratings_1)
    recommender.update(ratings=train_ratings_2)
    recs, _ = recommender.recommend(all_contexts, 1)
    num_hits = sum((user_id, rec) in test_ratings for user_id, rec in zip(users, recs[:, 0]))
    hit_rate2 = num_hits / NUM_USERS_ML100K

    # The hit rate should have increased.
    assert hit_rate1 < hit_rate2, hit_rate2


def test_recommend_simple(recommender):
    """Test that recommender will recommend reasonable items in simple setting."""
    users = {0: np.zeros((0,)),
             1: np.zeros((0,))}
    items = {0: np.zeros((0,)),
             1: np.zeros((0,)),
             2: np.zeros((0,))}
    assert NUM_USERS_SIMPLE == len(users)
    assert NUM_ITEMS_SIMPLE == len(items)
    ratings = {(0, 0): (5, np.zeros((0,))),
               (0, 1): (1, np.zeros((0,))),
               (0, 2): (5, np.zeros((0,))),
               (1, 0): (5, np.zeros((0,)))}
    recommender.reset(users, items, ratings)
    user_contexts = collections.OrderedDict([(1, np.zeros((0,)))])
    recs, _ = recommender.recommend(user_contexts, 1)
    recommender.predict([(1, 1, np.zeros(0,)), (1, 2, np.zeros(0,))])
    assert recs.shape == (1, 1)
    # The recommender should have recommended the item that user0 rated the highest.
    assert recs[0, 0] == 2


def rmse(predictions, targets):
    """Compute the root mean squared error (RMSE) between prediction and target vectors."""
    return np.sqrt(((predictions - targets) ** 2).mean())


def mock_select_online_users(self):
    """Return the users online at a given timestep.

    This functions is meant to replace the _select_online_users method in an environment
    when used for testing.
    """
    # pylint: disable=protected-access
    num_online = int(len(self._users) * self._rating_frequency)
    start_id = (num_online * (self._timestep + 1)) % len(self._users)
    end_id = min(start_id + num_online, len(self._users))
    return np.arange(start_id, end_id)
