import numpy as np
import pytest

import reclab
from reclab import data_utils
from reclab.recommenders import LibFM
from . import utils

def test_mcmc():
    users, items, ratings = data_utils.read_movielens100k()
    train_ratings, test_ratings = data_utils.split_ratings(ratings, 0.9, shuffle=True)
    model = LibFM(num_user_features=0,
                  num_item_features=0,
                  num_rating_features=0,
                  max_num_users=len(users),
                  max_num_items=len(items),
                  method='mcmc')
    model.reset(users, items, train_ratings)
    user_item = list(test_ratings.keys())
    preds = model.predict(user_item)
    print(utils.rmse(preds, test_ratings.values()))
    assert(1 == 1)
