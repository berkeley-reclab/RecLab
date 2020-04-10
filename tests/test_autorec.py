"""Tests for the Autorec recommender."""
from reclab.recommenders import Autorec
from . import utils


def test_predict():
    """Test that Autorec predicts well and that it gets better with more data."""
    recommender = Autorec(utils.NUM_USERS_ML100K, utils.NUM_ITEMS_ML100K,
                          hidden_neuron=500, lambda_value=1,
                          train_epoch=50, batch_size=20, optimizer_method='Adam',
                          grad_clip=False, base_lr=1e-2, decay_epoch_step=int(500), display_step=20)
    utils.test_predict_ml100k(recommender, rmse_threshold=1.5)


def test_recommend():
    """Test that Autorec will recommend reasonable items."""
    recommender = Autorec(utils.NUM_USERS_SIMPLE, utils.NUM_ITEMS_SIMPLE,
                          hidden_neuron=200, lambda_value=1,
                          train_epoch=50, batch_size=20, optimizer_method='Adam',
                          grad_clip=False, base_lr=1e-3, decay_epoch_step=int(500), display_step=20)
    utils.test_recommend_simple(recommender)
