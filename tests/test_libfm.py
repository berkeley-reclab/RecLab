import numpy as np
import pytest

import reclab
from reclab import data_utils
from reclab.recommenders import LibFM
from . import utils

def test_sgd():
    users, items, ratings = data_utils.read_movielens100k()
    train_ratings, test_ratings = data_utils.split_ratings(ratings, 0.9, shuffle=True)
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
                  num_iter=128)
    model.reset(users, items, train_ratings)
    user_item = [(key[0], key[1], val[1]) for key, val in test_ratings.items()]
    preds = model.predict(user_item)
    targets = [t[0] for t in test_ratings.values()]
    assert utils.rmse(preds, targets) < 1.0
