"""Tests for the Autorec recommender."""
from reclab.recommenders import Autorec
from . import utils


def test_predict():
    """Test that Autorec predicts well and that it gets better with more data."""
    recommender = Autorec(utils.NUM_USERS_ML100K,
                          utils.NUM_ITEMS_ML100K,
                          hidden_neuron=500,
                          lambda_value=20,
                          train_epoch=50,
                          batch_size=20,
                          grad_clip=False,
                          base_lr=1e-4,
                          random_seed=0)
    utils.test_predict_ml100k(recommender, rmse_threshold=1.3)


def test_recommend():
    """Test that Autorec will recommend reasonable items."""
    recommender = Autorec(utils.NUM_USERS_SIMPLE,
                          utils.NUM_ITEMS_SIMPLE,
                          hidden_neuron=500,
                          lambda_value=20,
                          train_epoch=1000,
                          batch_size=20,
                          grad_clip=False,
                          base_lr=1e-4,
                          random_seed=0)
    utils.test_recommend_simple(recommender)
