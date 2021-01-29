"""Tests for the CFNADE recommender."""
from reclab.recommenders.cfnade import Cfnade
from . import utils


def test_cfnade_predict():
    """Test that CFNADE predicts well and that it gets better with more data."""
    recommender = Cfnade(num_users=utils.NUM_USERS_ML100K,
                         num_items=utils.NUM_ITEMS_ML100K,
                         batch_size=64,
                         train_epoch=10,
                         rating_bucket=5,
                         hidden_dim=250,
                         learning_rate=0.001,
                         random_seed=0)
    utils.test_predict_ml100k(recommender, rmse_threshold=1.2)


def test_cfnade_recommend():
    """Test that CFNADE will recommend reasonable items."""
    recommender = Cfnade(num_users=utils.NUM_USERS_SIMPLE,
                         num_items=utils.NUM_ITEMS_SIMPLE,
                         batch_size=1,
                         train_epoch=10,
                         rating_bucket=5,
                         hidden_dim=250,
                         learning_rate=0.001,
                         random_seed=0)
    utils.test_recommend_simple(recommender)
