"""Implementation of the CF-NADE recommender using Keras."""
from keras.layers import Input, Dropout, Lambda, add
from keras.models import Model
import keras.regularizers
from keras.optimizers import Adam
import numpy as np

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
            rating_bucket=5, hidden_dim=500,
            learning_rate=0.001, normalized_layer=False,
            random_seed=0):
        """Create new Cfnade recommender."""
        super().__init__()
        self._num_users = num_users
        self._num_items = num_items
        self._batch_size = batch_size
        if num_items <= batch_size:
            self._batch_size = num_items
        self._input_dim0 = num_users
        self._rating_bucket = rating_bucket
        self._rate_score = np.array(np.arange(1, rating_bucket+1), np.float32)
        self._hidden_dim = hidden_dim
        self._learning_rate = learning_rate
        self._train_epoch = train_epoch
        self._hyperparameters.update(locals())
        self._new_items = np.zeros(num_items)
        np.random.seed(random_seed)

        # We only want the function arguments so remove class related objects.
        del self._hyperparameters['self']
        del self._hyperparameters['__class__']

        # Prepare model
        input_layer = Input(shape=(self._input_dim0, self._rating_bucket), name='input_ratings')
        output_ratings = Input(shape=(self._input_dim0, self._rating_bucket), name='output_ratings')
        input_masks = Input(shape=(self._input_dim0,), name='input_masks')
        output_masks = Input(shape=(self._input_dim0,), name='output_masks')
        nade_layer = Dropout(0.0)(input_layer)
        nade_layer = NADE(
                        hidden_dim=self._hidden_dim, activation='tanh', bias=True,
                        W_regularizer=keras.regularizers.l2(0.02),
                        V_regularizer=keras.regularizers.l2(0.02),
                        b_regularizer=keras.regularizers.l2(0.02),
                        c_regularizer=keras.regularizers.l2(0.02),
                        normalized_layer=normalized_layer)(nade_layer)

        predicted_ratings = Lambda(
            utils.prediction_layer,
            output_shape=utils.prediction_output_shape,
            name='predicted_ratings')(nade_layer)

        func_d = Lambda(
            utils.d_layer, output_shape=utils.d_output_shape,
            name='func_d')(input_masks)
        sum_masks = add([input_masks, output_masks])
        func_d_2 = Lambda(
            utils.D_layer, output_shape=utils.D_output_shape,
            name='func_d_2')(sum_masks)
        loss_out = Lambda(
            utils.rating_cost_lambda_func, output_shape=(1, ),
            name='nade_loss')([nade_layer, output_ratings,
                               input_masks, output_masks, func_d_2, func_d])

        self._cf_nade_model = Model(
            inputs=[input_layer, output_ratings, input_masks, output_masks],
            outputs=[loss_out, predicted_ratings])
        optimizer = Adam(self._learning_rate, 0.9, 0.999, 1e-8)
        self._cf_nade_model.compile(
            loss={'nade_loss': lambda y_true, y_pred: y_pred},
            optimizer=optimizer)
        self._cf_nade_model.save_weights('model.h5')

    @property
    def name(self):  # noqa: D102
        return 'cfnade'

    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        super().update(users, items, ratings)
        self._cf_nade_model.load_weights('model.h5')

        ratings_matrix = self._ratings.toarray()
        ratings_matrix = np.around(ratings_matrix.transpose())
        ratings_matrix = ratings_matrix.astype(int)

        train_set = utils.DataSet(ratings_matrix,
                                  num_users=self._num_users,
                                  num_items=self._num_items,
                                  batch_size=self._batch_size,
                                  rating_bucket=self._rating_bucket,
                                  mode=0)
        self._cf_nade_model.fit_generator(train_set.generate(),
                                          steps_per_epoch=(self._num_items // self._batch_size),
                                          epochs=self._train_epoch,
                                          callbacks=[train_set], verbose=1)

    def _predict(self, user_item):  # noqa: D102
        ratings_matrix = self._ratings.toarray()
        ratings_matrix = np.around(ratings_matrix.transpose())
        ratings_matrix = ratings_matrix.astype(int)

        # keep track of unseen items in ratings
        ratings_matrix_total = ratings_matrix.transpose().sum(axis=1)
        self._new_items = np.where(ratings_matrix_total == 0)[0]

        test_set = utils.DataSet(ratings_matrix,
                                 num_users=self._num_users,
                                 num_items=self._num_items,
                                 batch_size=self._batch_size,
                                 rating_bucket=self._rating_bucket,
                                 mode=2)
        pred_rating = []
        for batch in test_set.generate():
            pred_matrix = self._cf_nade_model.predict(batch[0])[1]
            pred_rating_batch = pred_matrix * self._rate_score[np.newaxis, np.newaxis, :]
            pred_rating_batch = pred_rating_batch.sum(axis=2)
            pred_rating.append(pred_rating_batch)
        pred_rating = np.concatenate(pred_rating, axis=0)

        predictions = []
        for user, item, _ in user_item:
            if item in self._new_items:
                predictions.append(3)
            else:
                predictions.append(pred_rating[item, user])

        return np.array(predictions)
