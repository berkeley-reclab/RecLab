"""Tests for the LLORMA recommender."""
from reclab.recommenders.llorma import Llorma
from . import utils


def test_llorma_predict():
    """Test that LLORMA predicts well and that it gets better with more data."""
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
    utils.test_predict_ml100k(recommender, rmse_threshold=1.1)


def test_llorma_recommend():
    """Test that LLORMA will recommend reasonable items."""
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
    utils.test_recommend_simple(recommender)
