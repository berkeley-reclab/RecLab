"""Tests for the LibFM recommender."""
from reclab.recommenders import LibFM
from . import utils


def test_sgd_predict():
    """Test that LibFM trained with SGD predicts well and that it gets better with more data."""
    recommender = LibFM(num_user_features=0,
                        num_item_features=0,
                        num_rating_features=0,
                        max_num_users=utils.NUM_USERS_ML100K,
                        max_num_items=utils.NUM_ITEMS_ML100K,
                        method='sgd',
                        learning_rate=0.003,
                        num_two_way_factors=8,
                        bias_reg=0.04,
                        one_way_reg=0.04,
                        two_way_reg=0.04,
                        num_iter=128,
                        seed=0)
    utils.test_predict_ml100k(recommender, rmse_threshold=1.1)


def test_sgd_recommend():
    """Test that LibFM trained with SGD will recommend reasonable items."""
    recommender = LibFM(num_user_features=0,
                        num_item_features=0,
                        num_rating_features=0,
                        max_num_users=utils.NUM_USERS_ML100K,
                        max_num_items=utils.NUM_ITEMS_ML100K,
                        method='sgd',
                        learning_rate=0.01,
                        num_two_way_factors=8,
                        num_iter=128,
                        seed=0)
    utils.test_recommend_simple(recommender)
