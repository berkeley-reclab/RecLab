"""Implementation of cfnade recommender using Keras."""
import time
import numpy as np

from keras.layers import Input, Dropout, Lambda, add
from keras.models import Model
import keras.regularizers
from keras.optimizers import Adam

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
    input_dim1: int
        input dimension to construct the layer
    hidden_dim: int
        hidden dimension to construct the layer
    learning_rate: float
        learning rate
    cf_nade_model: Keras Model object
        Keras model for predictions
    optimizer: Adam
        optimizer for CFNade is Adam
    """

    def __init__(
            self, num_users, num_items,
            train_set=None, batch_size=64, train_epoch=10,
            input_dim1=5, hidden_dim=250,
            learning_rate=0.001):
        """Create new Cfnade recommender."""
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.batch_size = batch_size
        self.num_batch = self.num_items//self.batch_size
        self.train_set = train_set
        self.input_dim0 = self.num_users
        self.input_dim1 = input_dim1
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.train_epoch = train_epoch
        self.train_rmse_callback = utils.RMSE_eval(data_set=self.train_set, training_set=True)
        # Prepare model
        input_layer = Input(shape=(self.input_dim0, self.input_dim1), name='input_ratings')
        output_ratings = Input(shape=(self.input_dim0, self.input_dim1), name='output_ratings')
        input_masks = Input(shape=(self.input_dim0,), name='input_masks')
        output_masks = Input(shape=(self.input_dim0,), name='output_masks')
        nade_layer = Dropout(0.0)(input_layer)
        nade_layer = NADE(
                        hidden_dim=self.hidden_dim, activation='tanh', bias=True,
                        W_regularizer=keras.regularizers.l2(0.02),
                        V_regularizer=keras.regularizers.l2(0.02),
                        b_regularizer=keras.regularizers.l2(0.02),
                        c_regularizer=keras.regularizers.l2(0.02))(nade_layer)

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

        self.cf_nade_model = Model(
            inputs=[input_layer, output_ratings, input_masks, output_masks],
            outputs=[loss_out, predicted_ratings])
        self.optimizer = Adam(self.learning_rate, 0.9, 0.999, 1e-8)
        self.cf_nade_model.compile(
            loss={'nade_loss': lambda y_true, y_pred: y_pred},
            optimizer=self.optimizer)

    def update(self, users=None, items=None, ratings=None):
        """Update the recommender with new user, item, and rating data.

        Parameters
        ----------
        users : dict, optional
            The new users where the key is the user id while the value is the
            user features.
        items : dict, optional
            The new items where the key is the user id while the value is the
            item features.
        ratings : dict, optional
            The new ratings where the key is a double whose first index is the
            id of the user making the rating and the second index is the id of the item being
            rated. The value is a double whose first index is the rating value and the second
            index is a numpy array that represents the context in which the rating was made.
        """
        super().update(users, items, ratings)
        ratings_matrix = self._ratings.toarray()
        ratings_matrix = np.around(ratings_matrix.transpose())
        self.train_set = ratings_matrix.astype(int)

        # Training
        start_time = time.time()
        self.cf_nade_model.fit_generator(
            utils.data_gen(self.train_set, self.batch_size, self.num_users, 0),
            steps_per_epoch=(self.num_items//self.batch_size),
            epochs=self.train_epoch,
            callbacks=[self.train_set, self.train_rmse_callback], verbose=1)
        print('Training //', 'Epochs %d //' % (self.train_epoch))
        print('Elapsed time : %d sec' % (time.time() - start_time))

    def _predict(self, user_item, round_rat=False):
        """Predict the ratings of user-item pairs. This internal version assumes inner ids.

        Parameters
        ----------
        user_item : list of tuple
            The list of all user-item pairs along with the rating context.
            Each element is a triple where the first element in the tuple is
            the inner user id, the second element is the inner item id and the third element
            is the context in which the item will be rated.
        round_rat : bool
            Cfnade predicts ratings as continuous. Set to true to round to integers.

        Returns
        -------
        predictions : np.ndarray
            The rating predictions where predictions[i] is the prediction of the i-th pair.
        """
        users = [triple[0] for triple in user_item]
        items = [triple[1] for triple in user_item]
        user_item = zip(users, items)
        rate_score = np.array([1, 2, 3, 4, 5], np.float32)
        test_df = np.zeros((self.num_items, self.num_users, 5))
        pred_rating = np.empty((0, 0))
        for i, batch in enumerate(utils.data_gen(test_df, self.batch_size, len(users), 2)):
            pred_matrix = self.cf_nade_model.predict(batch[0])[1]
            pred_rating_batch = (pred_matrix * rate_score[np.newaxis, np.newaxis, :]).sum(axis=2)
            pred_rating = np.append(pred_rating, pred_rating_batch, axis=0)

        predictions = np.ndarray(shape=(1, len(users)), dtype=float)
        for i, (user, item) in user_item:
            predictions[i] = pred_rating[item, user]

        if round_rat:
            predictions = predictions.astype(int)
        return predictions
