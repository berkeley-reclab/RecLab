"""Tests for the CFNADE recommender."""
import collections

import numpy as np

from reclab import data_utils
from reclab.recommenders.cfnade import Cfnade
from . import utils


def test_cfnade_predict():
    """Test that CFNADE predicts well and that it gets better with more data."""
    users, items, ratings = data_utils.read_dataset(name='ml-100k')
    train_ratings, test_ratings = data_utils.split_ratings(ratings, 0.9, shuffle=True)
    train_ratings_1, train_ratings_2 = data_utils.split_ratings(train_ratings, 0.5)
    recommender = Cfnade(num_users=len(users),
                         num_items=len(items),
                         batch_size=64,
                         train_epoch=10,
                         rating_bucket=5,
                         hidden_dim=250,
                         learning_rate=0.001)
    print("Reset")
    recommender.reset(users, items, train_ratings_1)
    user_item = [(key[0], key[1], val[1]) for key, val in test_ratings.items()]
    print("Predict")
    preds = recommender.predict(user_item)
    targets = [t[0] for t in test_ratings.values()]
    rmse1 = utils.rmse(preds, targets)
    # We should get a relatively low RMSE here.
    assert rmse1 < 1.2

    recommender.update(ratings=train_ratings_2)
    preds = recommender.predict(user_item)
    rmse2 = utils.rmse(preds, targets)

    # The RMSE should have reduced.
    assert rmse1 > rmse2


def test_cfnade_recommend():
    """Test that CFNADE will recommend reasonable items."""
    users = {0: np.zeros((0,)),
             1: np.zeros((0,))}
    items = {0: np.zeros((0,)),
             1: np.zeros((0,)),
             2: np.zeros((0,))}
    ratings = {(0, 0): (5, np.zeros((0,))),
               (0, 1): (1, np.zeros((0,))),
               (0, 2): (5, np.zeros((0,))),
               (1, 0): (5, np.zeros((0,)))}
    recommender = Cfnade(num_users=len(users),
                         num_items=len(items),
                         batch_size=64,
                         train_epoch=10,
                         rating_bucket=5,
                         hidden_dim=250,
                         learning_rate=0.001)
    recommender.reset(users, items, ratings)
    user_contexts = collections.OrderedDict([(1, np.zeros((0,)))])
    recs, _ = recommender.recommend(user_contexts, 1)
    recommender.predict([(1, 1, np.zeros(0,)), (1, 2, np.zeros(0,))])
    assert recs.shape == (1, 1)
    # The recommender should have recommended the item that user0 rated the highest.
    assert recs[0, 0] == 2
