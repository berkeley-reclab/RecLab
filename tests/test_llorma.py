"""Tests for the LLORMA recommender."""
import collections

import numpy as np

from reclab import data_utils
from reclab.recommenders.llorma import Llorma
from . import utils


def test_llorma_predict():
    """Test that LLORMA predicts well and that it gets better with more data."""
    users, items, ratings = data_utils.read_movielens100k()
    train_ratings, test_ratings = data_utils.split_ratings(ratings, 0.9, shuffle=True)
    train_ratings_1, train_ratings_2 = data_utils.split_ratings(train_ratings, 0.5)
    recommender = Llorma(n_anchor=10,
                         pre_rank=5,
                         pre_learning_rate=2e-4,
                         pre_lambda_val=10,
                         pre_train_steps=100,
                         rank=10,
                         learning_rate=1e-2,
                         lambda_val=1e-3,
                         train_steps=10,
                         batch_size=128,
                         use_cache=False,
                         result_path='results')
    recommender.reset(users, items, train_ratings_1)
    user_item = [(key[0], key[1], val[1]) for key, val in test_ratings.items()]
    preds = recommender.predict(user_item)
    targets = [t[0] for t in test_ratings.values()]
    rmse1 = utils.rmse(preds, targets)
    # We should get a relatively low RMSE here.
    assert rmse1 < 1.1

    recommender.update(ratings=train_ratings_2)
    preds = recommender.predict(user_item)
    rmse2 = utils.rmse(preds, targets)

    # The RMSE should have reduced.
    assert rmse1 > rmse2


def test_llorma_recommend():
    """Test that LLORMA will recommend reasonable items."""
    users = {0: np.zeros((0,)),
             1: np.zeros((0,))}
    items = {0: np.zeros((0,)),
             1: np.zeros((0,)),
             2: np.zeros((0,))}
    ratings = {(0, 0): (5, np.zeros((0,))),
               (0, 1): (1, np.zeros((0,))),
               (0, 2): (5, np.zeros((0,))),
               (1, 0): (5, np.zeros((0,)))}
    recommender = Llorma(n_anchor=2,
                         pre_rank=5,
                         pre_learning_rate=2e-4,
                         pre_lambda_val=10,
                         pre_train_steps=100,
                         rank=10,
                         learning_rate=1e-2,
                         lambda_val=1e-3,
                         train_steps=10,
                         batch_size=128,
                         use_cache=False,
                         result_path='results')
    recommender.reset(users, items, ratings)
    user_contexts = collections.OrderedDict([(1, np.zeros((0,)))])
    recs, _ = recommender.recommend(user_contexts, 1)
    recommender.predict([(1, 1, np.zeros(0,)), (1, 2, np.zeros(0,))])
    assert recs.shape == (1, 1)
    # The recommender should have recommended the item that user0 rated the highest.
    assert recs[0, 0] == 2