"""Implementation of cfnade recommender using Keras."""
import time

import numpy as np
from keras.layers import Input, Dropout, Lambda, add
from keras.models import Model
import keras.regularizers
from keras.optimizers import Adam
from keras.callbacks import Callback

from .cfnade_lib.nade import NADE
from .cfnade_lib import utils
from .. import recommender


class Cfnade(recommender.PredictRecommender):
    """
    A Neural Autoregressive Distribution Estimator (NADE) for collaborative filtering (CF) tasks.

    Parameters
    ---------
    num_users : int
        Number of users in the environment.
    num_items : int
        Number of items in the environment.
    train_set : np.matrix
        Matrix of shape (num_users, num_items) populated with user ratings.
    train_epoch : int
        Number of epochs to train for each call.
    batch_size : int
        Batch size during initial training phase.
    rating_bucket: int
        number of rating buckets
    rate_score: array of float
        An array of corresponding rating score for each bucket
    hidden_dim: int
        hidden dimension to construct the layer
    learning_rate: float
        learning rate
    """

    def __init__(
            self, num_users, num_items,
            batch_size=64, train_epoch=10,
            rating_bucket=5, hidden_dim=250,
            learning_rate=0.001):
        """Create new Cfnade recommender."""
        print("Initialize...")
        super().__init__()
        self._num_users = num_users
        self._num_items = num_items
        self._batch_size = batch_size
        self._input_dim0 = num_users
        self._rating_bucket = rating_bucket
        self._rate_score = np.array(np.arange(1,rating_bucket+1), np.float32)
        self._hidden_dim = hidden_dim
        self._learning_rate = learning_rate
        self._train_epoch = train_epoch
        # Prepare model
        self.input_layer = Input(shape=(self._input_dim0, self._rating_bucket), name='input_ratings')
        self.output_ratings = Input(shape=(self._input_dim0, self._rating_bucket), name='output_ratings')
        self.input_masks = Input(shape=(self._input_dim0,), name='input_masks')
        self.output_masks = Input(shape=(self._input_dim0,), name='output_masks')
        self.nade_layer = Dropout(0.0)(self.input_layer)
        self.nade_layer = NADE(
                        hidden_dim=self._hidden_dim, activation='tanh', bias=True,
                        W_regularizer=keras.regularizers.l2(0.02),
                        V_regularizer=keras.regularizers.l2(0.02),
                        b_regularizer=keras.regularizers.l2(0.02),
                        c_regularizer=keras.regularizers.l2(0.02))(self.nade_layer)

        self.predicted_ratings = Lambda(
            utils.prediction_layer,
            output_shape=utils.prediction_output_shape,
            name='predicted_ratings')(self.nade_layer)

        self.func_d = Lambda(
            utils.d_layer, output_shape=utils.d_output_shape,
            name='func_d')(self.input_masks)
        self.sum_masks = add([self.input_masks, self.output_masks])
        self.func_d_2 = Lambda(
            utils.D_layer, output_shape=utils.D_output_shape,
            name='func_d_2')(self.sum_masks)
        self.loss_out = Lambda(
            utils.rating_cost_lambda_func, output_shape=(1, ),
            name='nade_loss')([self.nade_layer, self.output_ratings,
                               self.input_masks, self.output_masks, self.func_d_2, self.func_d])

        self.cf_nade_model = Model(
            inputs=[self.input_layer, self.output_ratings, self.input_masks, self.output_masks],
            outputs=[self.loss_out, self.predicted_ratings])
        self.optimizer = Adam(self._learning_rate, 0.9, 0.999, 1e-8)
        self.cf_nade_model.compile(
            loss={'nade_loss': lambda y_true, y_pred: y_pred},
            optimizer=self.optimizer)
        self.cf_nade_model.summary()

    def update(self, users=None, items=None, ratings=None):
        super().update(users, items, ratings)

        ratings_matrix = self._ratings.toarray()
        ratings_matrix = np.around(ratings_matrix.transpose())
        ratings_matrix = ratings_matrix.astype(int)
        train_set = utils.DataSet(ratings_matrix,
        num_users=self._num_users,
        num_items=self._num_items,
        batch_size=self._batch_size,
        rating_bucket=self._rating_bucket,
        mode=0)

        train_rmse_callback = utils.RMSE_eval(data_set=train_set, training_set=True, rate_score=self._rate_score)

        # Training
        print("Training...")
        print("items: ", self._num_items, "users: ", self._num_users)
        print("batch size: ", self._batch_size, "epochs: ", self._train_epoch)
        print("rating shape", ratings_matrix.shape)
        start_time = time.time()
        self.cf_nade_model.fit_generator(
            train_set.generate(),
            steps_per_epoch=(self._num_items//self._batch_size),
            epochs=self._train_epoch,
            callbacks=[train_set], verbose=1)
        print('Elapsed time : %d sec' % (time.time() - start_time))

    def _predict(self, user_item):
        users = [triple[0] for triple in user_item]
        items = [triple[1] for triple in user_item]
        user_item = zip(users, items)
        test_df = np.zeros((self._num_items, self._num_users, 5))
        test_set = utils.DataSet(test_df,
        num_users=self._num_users,
        num_items=self._num_items,
        batch_size=self._batch_size,
        rating_bucket=self._rating_bucket,
        mode=2)
        pred_rating = []
        print("Predicting...")
        for i, batch in enumerate(test_set.generate()):
            pred_matrix = self.cf_nade_model.predict(batch[0])[1]
            pred_rating_batch = (pred_matrix * self._rate_score[np.newaxis, np.newaxis, :]).sum(axis=2)
            pred_rating.append(pred_rating_batch)
        pred_rating = np.concatenate(pred_rating, axis=0)
        predictions = np.ndarray(shape=(1, len(users)), dtype=float)
        for i, (user, item) in user_item:
            predictions[i] = pred_rating[item, user]
