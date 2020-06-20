import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
sys.path.append('experiments')
sys.path.append('.')
sys.path.append('experiments/experiments')
from env_defaults import TOPICS_STATIC_SMALL, get_len_trial
from llorma_optimal_params import OPT_TOPICS_SMALL
from run_utils import get_env_dataset, run_env_experiment
from reclab.environments import Topics
from reclab.recommenders import Llorma


# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'Mihaela'
overwrite = True


ENV_DICT = TOPICS_STATIC_SMALL
len_trial = get_len_trial(ENV_DICT)
trial_seeds = [0]

# Environment setup
environment_name = ENV_DICT['name']
# env = LatentFactorBehavior(**ENV_DICT['params'], **ENV_DICT['optional_params'])
env = Topics(**ENV_DICT['params'], **ENV_DICT['optional_params'])

# Recommender setup
recommender_name = 'Llorma-test'
recommender_class = Llorma

recommender = recommender_class(max_user=ENV_DICT['params']['num_users'],
                                max_item=ENV_DICT['params']['num_items'],
                                **OPT_TOPICS_SMALL,
                                )

for i, seed in enumerate(trial_seeds):
    run_env_experiment(
            [env],
            [recommender],
            [seed],
            1,
            environment_names=[environment_name],
            recommender_names=[recommender_name],
            bucket_name=bucket_name,
            data_dir=data_dir,
            overwrite=overwrite)
