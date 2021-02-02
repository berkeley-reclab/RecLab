import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from env_defaults import *
from experiment import get_env_dataset
from tuner import ModelTuner
from reclab.environments import Topics
from reclab.recommenders import LibFM

def test_offline(user_dist_choice, initial_sampling, use_mse):
    # Environment setup
    environment_name = TOPICS_DYNAMIC['name']
    env = Topics(**TOPICS_DYNAMIC['params'], **TOPICS_DYNAMIC['optional_params'],
                 user_dist_choice=user_dist_choice, initial_sampling=initial_sampling)
    env.seed(0)

    # Recommender setup
    recommender_class = LibFM


    # ====Step 5====
    starting_data = get_env_dataset(env)


    # ====Step 6====
    # Recommender tuning setup
    n_fold = 5
    num_users, num_items = get_num_users_items(TOPICS_DYNAMIC)
    default_params = dict(num_user_features=0,
                          num_item_features=0,
                          num_rating_features=0,
                          max_num_users=num_users,
                          max_num_items=num_items,
                          method='mcmc')
    tuner = ModelTuner(starting_data,
                       default_params,
                       recommender_class,
                       n_fold=n_fold,
                       verbose=True,
                       use_mse=use_mse)

    # Tune the performance independent hyperparameters.
    init_stdev = [1.0]
    num_iter = [200]
    num_two_way_factors = [20]

    results = tuner.evaluate_grid(
        init_stdev=init_stdev,
        num_iter=num_iter,
        num_two_way_factors=num_two_way_factors)

test_offline('powerlaw', 'powerlaw', False)
test_offline('powerlaw', 'powerlaw', True)
test_offline('uniform', 'uniform', False)
test_offline('uniform', 'uniform', True)
