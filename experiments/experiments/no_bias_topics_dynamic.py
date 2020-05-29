import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
sys.path.append('experiments')
sys.path.append('.')
sys.path.append('experiments/experiments')
from env_defaults import TOPICS_DYNAMIC, get_len_trial
from run_utils import get_env_dataset, run_env_experiment
from reclab.environments import Topics
from reclab.recommenders import LibFM

bucket_name = 'recsys-eval'
data_dir = 'master'
overwrite = True

ENV_DICT = TOPICS_DYNAMIC
ENV = Topics
# Experiment setup.
num_users = ENV_DICT['params']['num_users']
num_init_ratings = ENV_DICT['optional_params']['num_init_ratings']
num_final_ratings = ENV_DICT['misc']['num_final_ratings']
rating_frequency = ENV_DICT['optional_params']['rating_frequency']

trial_seeds = [0]
len_trial = get_len_trial(ENV_DICT)

# Environment setup
environment_name = ENV_DICT['name']
env = ENV(**ENV_DICT['params'], **ENV_DICT['optional_params'])

# Recommender setup
recommender_name = 'LibFM (no bias)'
recommender_class = LibFM


DEFAULT_PARAMS = dict(num_user_features=0,
                      num_item_features=0,
                      num_rating_features=0,
                      max_num_users=num_users,
                      max_num_items=ENV_DICT['params']['num_items'],
                      method='sgd',
                      use_global_bias=False,
                      use_one_way=False,
                      num_two_way_factors = 20,
                      num_iter = 100,
                      init_stdev = 1,
                      reg = 0,
                      learning_rate = 0.03,
                      )


recommender = recommender_class(**DEFAULT_PARAMS)

for i, seed in enumerate(trial_seeds):
    run_env_experiment(
            [env],
            [recommender],
            [seed],
            len_trial,
            environment_names=[environment_name],
            recommender_names=[recommender_name],
            bucket_name=bucket_name,
            data_dir=data_dir,
            overwrite=overwrite)
