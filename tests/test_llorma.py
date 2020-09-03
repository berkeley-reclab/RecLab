"""Tests for the LLORMA recommender."""
from reclab.recommenders.llorma import Llorma
from . import utils


def test_llorma_predict():
    """Test that LLORMA predicts well and that it gets better with more data."""
    recommender = Llorma(max_user=utils.NUM_USERS_ML100K,
                         max_item=utils.NUM_ITEMS_ML100K,
                         n_anchor=10,
                         pre_rank=5,
                         pre_learning_rate=1e-3,
                         pre_lambda_val=10,
                         pre_train_steps=10,
                         rank=10,
                         learning_rate=1e-3,
                         lambda_val=1e-3,
                         train_steps=10,
                         batch_size=1024,
                         use_cache=True,
                         result_path='results',
                         random_seed=0)
    utils.test_predict_ml100k(recommender, rmse_threshold=1.1)


def test_llorma_recommend():
    """Test that LLORMA will recommend reasonable items."""
    recommender = Llorma(max_user=utils.NUM_USERS_SIMPLE,
                         max_item=utils.NUM_ITEMS_SIMPLE,
                         n_anchor=2,
                         pre_rank=5,
                         pre_learning_rate=1e-3,
                         pre_lambda_val=10,
                         pre_train_steps=10,
                         rank=10,
                         learning_rate=1e-2,
                         lambda_val=1e-3,
                         train_steps=10,
                         batch_size=128,
                         use_cache=True,
                         result_path='results',
                         random_seed=0)
    utils.test_recommend_simple(recommender)
