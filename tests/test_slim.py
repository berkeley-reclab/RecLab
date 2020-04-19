"""Tests for the SLIM recommender."""
from reclab.recommenders import SLIM
from . import utils


def test_predict():
    """Test that SLIM predicts well and that it gets better with more data."""
    recommender = SLIM(alpha=0.1, l1_ratio=1e-3, seed=0)
    utils.test_binary_recommend_ml100k(recommender, 0.1)


def test_recommend():
    """Test that SLIM will recommend reasonable items."""
    recommender = SLIM(alpha=0.1, l1_ratio=1e-3, seed=0)
    utils.test_recommend_simple(recommender)
