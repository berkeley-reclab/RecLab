"""Tests for the EASE recommender."""
from reclab.recommenders import EASE
from . import utils


def test_predict():
    """Test that EASE predicts well and that it gets better with more data."""
    recommender = EASE(lam=100, binarize=True)
    utils.test_binary_recommend_ml100k(recommender, 0.1)


def test_recommend():
    """Test that EASE will recommend reasonable items."""
    recommender = EASE(lam=100)
    utils.test_recommend_simple(recommender)
