"""Tests for the Autorec recommender."""
from reclab.recommenders import Autorec
from reclab import data_utils
import numpy as np
from tests import utils

recommender = Autorec(utils.NUM_USERS_ML10M, utils.NUM_ITEMS_ML10M,
                          hidden_neuron=500, lambda_value=1,
                          train_epoch=50, batch_size=20, optimizer_method='Adam',
                          grad_clip=False, base_lr=1e-2, decay_epoch_step=int(500), display_step=20)
users, items, ratings = data_utils.read_dataset('ml-10m')
print(len(users))
print(len(items))
train_ratings, test_ratings = data_utils.split_ratings(ratings, 0.9, shuffle=True, seed=0)
recommender.reset(users, items, train_ratings)
user_item = [(key[0], key[1], val[1]) for key, val in test_ratings.items()]
preds = recommender.predict(user_item)
targets = [t[0] for t in test_ratings.values()]
rmse = _rmse(preds, targets)

def _rmse(predictions, targets):
    """Compute the root mean squared error (RMSE) between prediction and target vectors."""
    return np.sqrt(((predictions - targets) ** 2).mean())
