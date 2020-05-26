import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
sys.path.append('experiments')
sys.path.append('.')
sys.path.append('experiments/experiments')
from env_defaults import LATENT_STATIC, get_len_trial
from llorma_optimal_params import OPT_LATENT
from run_utils import get_env_dataset, run_env_experiment
from reclab.environments import LatentFactorBehavior
from reclab.recommenders import Llorma


# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'master'
overwrite = True


ENV_DICT = LATENT_STATIC
len_trial = get_len_trial(ENV_DICT)
trial_seeds = [0]

# Environment setup
environment_name = ENV_DICT['name']
# env = LatentFactorBehavior(**ENV_DICT['params'], **ENV_DICT['optional_params'])
env = LatentFactorBehavior(**ENV_DICT['params'], **ENV_DICT['optional_params'])

# Recommender setup
recommender_name = 'Llorma'
recommender_class = Llorma

recommender = recommender_class(max_user=ENV_DICT['params']['num_users'],
                                 max_item=ENV_DICT['params']['num_items'],
                                 **OPT_LATENT)

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
