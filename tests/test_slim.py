"""Tests for the Autorec recommender."""
from reclab.recommenders import SLIM
from . import utils


def test_predict():
    """Test that Autorec predicts well and that it gets better with more data."""
    recommender = SLIM(alpha=0.1, l1_ratio=1e-3, seed=0)
    # We want a very high rmse threshold since SLIM doesn't try to drive RMSE down.
    utils.test_binary_recommend_ml100k(recommender, 0.1)

def test_recommend():
    """Test that Autorec will recommend reasonable items."""
    recommender = SLIM(alpha=0.1, l1_ratio=1e-3, seed=0)
    utils.test_recommend_simple(recommender)
