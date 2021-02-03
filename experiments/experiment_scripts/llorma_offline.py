import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from env_defaults import *
from experiment import get_env_dataset, sample_ratings
from tuner import ModelTuner
from reclab.environments import Topics
from reclab.recommenders import Llorma
from llorma_optimal_params import OPT_TOPICS

def test_offline(user_dist_choice, initial_sampling, use_mse, sample=False, low=False):
    # Environment setup
    environment_name = TOPICS_STATIC['name']
    env = Topics(**TOPICS_STATIC['params'], **TOPICS_STATIC['optional_params'],
                 user_dist_choice=user_dist_choice, initial_sampling=initial_sampling)
    env.seed(0)

    # Recommender setup
    recommender_name = 'Llorma'
    recommender_class = Llorma

    # ====Step 5====
    if sample:
        starting_data = sample_ratings(env, low_ratings=low)
    else:
        starting_data = get_env_dataset(env)



    # ====Step 6====
    # Recommender tuning setup
    n_fold = 5
    num_users, num_items = get_num_users_items(TOPICS_STATIC)
    default_params = dict()
    tuner = ModelTuner(starting_data,
                       default_params,
                       recommender_class,
                       n_fold=n_fold,
                       verbose=True,
                       use_mse=use_mse)
    params = {'max_user': TOPICS_STATIC['params']['num_users'],
              'max_item': TOPICS_STATIC['params']['num_items']}
    params.update(OPT_TOPICS)
    results = tuner.evaluate(params=params)

print('=========POWERLAW============')
test_offline('powerlaw', 'powerlaw', False)
test_offline('powerlaw', 'powerlaw', True)
print('=========UNIFORM============')
test_offline('uniform', 'uniform', False)
test_offline('uniform', 'uniform', True)
print('=========POWERLAW RATINGS HIGH============')
test_offline('uniform', 'uniform', False, True)
test_offline('uniform', 'uniform', True, True)
print('=========POWERLAW RATINGS LOW============')
test_offline('uniform', 'uniform', False, True, True)
test_offline('uniform', 'uniform', True, True, True)




