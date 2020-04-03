import glob
import os
import random
import numpy as np
import tensorflow as tf

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Lambda, add
from keras import backend as K
from keras.models import Model
from keras.callbacks import Callback
import keras.regularizers
from keras.optimizers import Adam

from nade import NADE
import utils


class CFNade():
    def __init__(
            self, batch_size,
            num_users, num_items, train_set, train_epoch,
            input_dim1, hidden_dim, std, alpha,
            data_sample, lr, beta_1, beta_2, epsilon, shuffle):

        self.num_users = num_users
        self.num_items = num_items
        self.batch_size = batch_size
        self.num_batch = self.num_items//self.batch_size
        self.train_set = train_set
        self.data_sample = data_sample
        self.input_dim0 = self.num_users
        self.input_dim1 = input_dim1
        self.hidden_dim = hidden_dim
        self.std = std
        self.alpha = alpha
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.train_epoch = train_epoch
        self.shuffle = shuffle
        self.train_rmse_callback = util.RMSE_eval(data_set=self.train_set, training_set=True)

    def prepare_model(self):
        input_layer = Input(shape=(self.input_dim0, self.input_dim1), name='input_ratings')
        output_ratings = Input(shape=(self.input_dim0, self.input_dim1), name='output_ratings')
        input_masks = Input(shape=(self.input_dim0,), name='input_masks')
        output_masks = Input(shape=(self.input_dim0,), name='output_masks')
        nade_layer = Dropout(0.0)(input_layer)
        nade_layer = NADE(
                        hidden_dim=hidden_dim, activation='tanh', bias=True,
                        W_regularizer=keras.regularizers.l2(0.02),
                        V_regularizer=keras.regularizers.l2(0.02),
                        b_regularizer=keras.regularizers.l2(0.02),
                        c_regularizer=keras.regularizers.l2(0.02))(nade_layer)

        predicted_ratings = Lambda(
                    utils.prediction_layer,
                    output_shape=utils.prediction_output_shape,
                    name='predicted_ratings')(nade_layer)

        d = Lambda(utils.d_layer, output_shape=utils.d_output_shape, name='d')(input_masks)
        sum_masks = add([input_masks, output_masks])
        D = Lambda(utils.D_layer, output_shape=utils.D_output_shape, name='D')(sum_masks)
        loss_out = Lambda(
                   utils.rating_cost_lambda_func, output_shape=(1, ),
                   name='nade_loss')([nade_layer, output_ratings, input_masks, output_masks, D, d])

        self.cf_nade_model = Model(
                         inputs=[input_layer, output_ratings, input_masks, output_masks],
                         outputs=[loss_out, predicted_ratings])
        self.optimizer = Adam(self.lr, self.beta_1, self.beta_2, self.epsilon)
        self.cf_nade_model.compile(
                     loss={'nade_loss': lambda y_true, y_pred: y_pred},
                     optimizer=adam)

    def train_model(self):
        start_time = time.time()
        self.cf_nade_model.fit_generator(
                   utils.data_gen(self.train_set, self.batch_size, 0),
                   steps_per_epoch=(self.num_items//self.batch_size),
                   epochs=self.train_epoch,
                   shuffle=self.shuffle,
                   callbacks=[self.train_set, self.train_rmse_callback], verbose=1)
        # unsure how to save the rmse from the callbacks, but they are displayed
        print("Training //", "Epochs %d //" % (self.train_epoch))
        print("Elapsed time : %d sec" % (time.time() - start_time))

    def predict(self, user_item):

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

        return pred_rating
