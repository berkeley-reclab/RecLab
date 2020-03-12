"""Tests for the LibFM recommender."""
import collections

import numpy as np

from reclab import data_utils
from reclab.recommenders import LibFM
from . import utils


def test_sgd_predict():
    """Test that LibFM trained with SGD predicts well and that it gets better with more data."""
    users, items, ratings = data_utils.read_movielens100k()
    train_ratings, test_ratings = data_utils.split_ratings(ratings, 0.9, shuffle=True)
    train_ratings_1, train_ratings_2 = data_utils.split_ratings(train_ratings, 0.5)
    model = LibFM(num_user_features=0,
                  num_item_features=0,
                  num_rating_features=0,
                  max_num_users=len(users),
                  max_num_items=len(items),
                  method='sgd',
                  learning_rate=0.003,
                  num_two_way_factors=8,
                  bias_reg=0.04,
                  one_way_reg=0.04,
                  two_way_reg=0.04,
                  num_iter=128,
                  seed=0)
    model.reset(users, items, train_ratings_1)
    user_item = [(key[0], key[1], val[1]) for key, val in test_ratings.items()]
    preds = model.predict(user_item)
    targets = [t[0] for t in test_ratings.values()]
    rmse1 = utils.rmse(preds, targets)
    # We should get a relatively low RMSE here.
    assert rmse1 < 1.1

    model.update(ratings=train_ratings_2)
    preds = model.predict(user_item)
    rmse2 = utils.rmse(preds, targets)

    # The RMSE should have reduced.
    assert rmse1 > rmse2


def test_sgd_recommend():
    """Test that LibFM trained with SGD will recommend reasonable items."""
    users = {0: np.zeros((0,)),
             1: np.zeros((0,))}
    items = {0: np.zeros((0,)),
             1: np.zeros((0,)),
             2: np.zeros((0,))}
    ratings = {(0, 0): (5, np.zeros((0,))),
               (0, 1): (1, np.zeros((0,))),
               (0, 2): (5, np.zeros((0,))),
               (1, 0): (5, np.zeros((0,)))}
    model = LibFM(num_user_features=0,
                  num_item_features=0,
                  num_rating_features=0,
                  max_num_users=len(users),
                  max_num_items=len(items),
                  method='sgd',
                  learning_rate=0.01,
                  num_two_way_factors=8,
                  num_iter=128,
                  seed=0)
    model.reset(users, items, ratings)
    user_contexts = collections.OrderedDict([(1, np.zeros((0,)))])
    recs, _ = model.recommend(user_contexts, 1)
    preds = model.predict([(1, 1, np.zeros(0,)), (1, 2, np.zeros(0,))])
    assert recs.shape == (1, 1)
    # The recommender should have recommended the item that user0 rated the highest.
    assert recs[0, 0] == 2
