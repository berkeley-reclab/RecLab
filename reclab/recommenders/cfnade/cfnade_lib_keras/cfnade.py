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

from data_gen import DataSet
from nade import NADE
import utils



class CFNade():

	def __init__(self, batch_size, 
					   num_users, num_items, data_sample = 1.0, input_dim0, 
					   input_dim1, hidden_dim, std, alpha):
		self.batch_size = batch_size
		self.num_users = num_users
		self.num_items = num_items
		self.data_sample = data_sample
		self.input_dim0 = input_dim0
		self.input_dim1 = input_dim1
		self.hidden_dim = hidden_dim
		self.std = std
		self.alpha = alpha

	def prepare_model(self):

		input_layer = Input(shape=(self.input_dim0,self.input_dim1),
		name='input_ratings')
		output_ratings = Input(shape=(self.input_dim0,self.input_dim1),
		name='output_ratings')
		input_masks = Input(shape=(self.input_dim0,),
		name='input_masks')
		output_masks = Input(shape=(self.input_dim0,),
		name='output_masks')

		nade_layer = Dropout(0.0)(input_layer)
		nade_layer = NADE(hidden_dim=hidden_dim,
		activation='tanh',
		bias=True,
		W_regularizer=keras.regularizers.l2(0.02),
		V_regularizer=keras.regularizers.l2(0.02),
		b_regularizer=keras.regularizers.l2(0.02),
		c_regularizer=keras.regularizers.l2(0.02))(nade_layer)

		predicted_ratings = Lambda(prediction_layer,
		output_shape=prediction_output_shape,
		name='predicted_ratings')(nade_layer)

		d = Lambda(d_layer,
			output_shape=d_output_shape,
			name='d')(input_masks)	

		sum_masks = add([input_masks, output_masks])
		D = Lambda(D_layer,
			output_shape=D_output_shape,
			name='D')(sum_masks)
		
		loss_out = Lambda(rating_cost_lambda_func,
			output_shape=(1,),
			name='nade_loss')([nade_layer,output_ratings,input_masks,output_masks,D,d])



