''' Keras implementation of cfnade recommender'''

import tensorflow as tf
from keras.models import Model

from .. import recommender
from .cfnade_lib import cfnade


class Cfnade(recommender.PredictRecommender):
    """
    A Neural Autoregressive Distribution Estimator (NADE) for collaborative filtering (CF) tasks

    Parameters
    ---------
    num_users : int
        Number of users in the environment.
    num_items : int
        Number of items in the environment.
    ratings : np.matrix
        Matrix of shape (num_users, num_items) populated with user ratings.
    train_epoch : int
        Number of epochs to train for each call.
    batch_size : int
        Batch size during initial training phase.
    
    """

	def __init__(self, num_users, num_items, 
                 train_set=None, batch_size=64, train_epoch=10,
                 input_dim1=5, hidden_dim=250, std=0.0, alpha=1.0, 
                 data_sample = 1.0, lr=0.001, beta_1=0.9, 
                 beta_2=0.999, epsilon=1e-8, shuffle=True):
        """Create new Cfnade recommender."""
        super().__init__()

        self.model = cfnade.CFNade(self, batch_size, 
                       num_users, num_items, train_set, train_epoch,
                       input_dim1, hidden_dim, std, alpha, 
                       data_sample, lr, beta_1, 
                       beta_2, epsilon, shuffle)

    def update(self, users=None, items=None, ratings=None):  
        super().update(users, items, ratings)
        
        ratings_matrix = self._ratings.toarray()
        ratings_matrix = np.around(ratings_matrix.transpose())
        ratings_matrix = ratings_matrix.astype(int)
        #one-hot encoding
        encoding = np.zeros((ratings_matrix.size, ratings_matrix.max() - ratings_matrix.min()+1))
        encoding[np.arange(ratings_matrix.size),ratings_matrix-1] = 1
        self.model.train_set = encoding

        self.model.prepare_model()
        self.model.train_model()

    def _predict(self, user_item, round_rat=False):
        """
        Predict items for user-item pairs.
        round_rat : bool
            Cfnade predicts ratings as continuous. Set to true to round to integers.
        """
        estimate = self.model.predict(user_item)
        if round_rat:
            estimate = estimate.astype(int)
        return estimate

    def reset(self, users=None, items=None, ratings=None):  
        self.model.prepare_model()
        super().reset(users, items, ratings)





