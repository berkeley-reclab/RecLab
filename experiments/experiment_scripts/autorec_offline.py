import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from env_defaults import *
from experiment import get_env_dataset
from tuner import ModelTuner
from reclab.environments import Topics
from reclab.recommenders import Autorec

def test_offline(user_dist_choice, initial_sampling, use_mse):
    # Environment setup
    environment_name = TOPICS_DYNAMIC['name']
    env = Topics(**TOPICS_DYNAMIC['params'], **TOPICS_DYNAMIC['optional_params'],
                 user_dist_choice=user_dist_choice, initial_sampling=initial_sampling)
    env.seed(0)

    # Recommender setup
    recommender_class = Autorec


    # ====Step 5====
    starting_data = get_env_dataset(env)


    # ====Step 6====
    # Recommender tuning setup
    n_fold = 5
    num_users, num_items = get_num_users_items(TOPICS_DYNAMIC)
    default_params = dict(num_users=num_users,
                          num_items=num_items)
    tuner = ModelTuner(starting_data,
                       default_params,
                       recommender_class,
                       n_fold=n_fold,
                       verbose=True,
                       use_mse=use_mse)

    # Tune the performance independent hyperparameters.
    train_epochs = [500]
    hidden_neurons = [500]
    lambda_values = [1.0]
    lrs = [0.001]
    lr_decay = [1.0]

    results = tuner.evaluate_grid(train_epoch=train_epochs,
                        hidden_neuron=hidden_neurons,
                        lambda_value=lambda_values,
                        base_lr=lrs,
                        lr_decay=lr_decay)

    recommender = recommender_class(train_epoch=1000,
                                    hidden_neuron=500,
                                    lambda_value=1.0,
                                    base_lr=0.001,
                                    **default_params)

test_offline('powerlaw', 'powerlaw', True)
test_offline('powerlaw', 'powerlaw', False)
test_offline('uniform', 'uniform', False)
test_offline('uniform', 'uniform', True)
