"""Tests for the KNN recommender."""
from reclab.recommenders import KNNRecommender
from . import utils


def test_user_predict():
    """Test that KNN-user predicts well and that it gets better with more data."""
    recommender = KNNRecommender(user_based=True)
    utils.test_predict_ml100k(recommender, rmse_threshold=1.1, test_dense=True)


def test_item_predict():
    """Test that KNN-user predicts well and that it gets better with more data."""
    recommender = KNNRecommender(user_based=False, shrinkage=0.1)
    utils.test_predict_ml100k(recommender, rmse_threshold=1.5, test_dense=True)


def test_user_recommend():
    """Test that KNN-item will recommend reasonable items."""
    recommender = KNNRecommender(user_based=True)
    utils.test_recommend_simple(recommender)


def test_item_recommend():
    """Test that KNN-item will recommend reasonable items."""
    recommender = KNNRecommender(user_based=True)
    utils.test_recommend_simple(recommender)
