import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
sys.path.append('experiments')
sys.path.append('.')
sys.path.append('experiments/experiments')
from env_defaults import LATENT_STATIC, ML_100K,  get_len_trial
from llorma_optimal_params import OPT_100K, LAMBDA_VAL, LEARNING_RATE
from run_utils import get_env_dataset, run_env_experiment
from reclab.recommenders import Llorma
from reclab.environments.latent_factors  import DatasetLatentFactor

#=====

i = 0

#=====

# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'Mihaela'
overwrite = True


ENV_DICT = ML_100K
len_trial = get_len_trial(LATENT_STATIC)
trial_seeds = [0]

# Environment setup
environment_name = ENV_DICT['name']
# env = LatentFactorBehavior(**ENV_DICT['params'], **ENV_DICT['optional_params'])
env = DatasetLatentFactor(**ENV_DICT['params'], **ENV_DICT['optional_params'])

# Recommender setup
recommender_name = ['Llorma-{}-{}'.format(i,j) for j in range(4)]
recommender_class = Llorma

print(LEARNING_RATE[0])

recommenders = [recommender_class(max_user=LATENT_STATIC['params']['num_users'],
                                  max_item=LATENT_STATIC['params']['num_items'],
                                  learning_rate=LEARNING_RATE[i],
                                  lambda_val=LAMBDA_VAL[j],
                                  **OPT_100K) for j in range(4)]

for j, recommender in enumerate(recommenders):
    run_env_experiment(
            [env],
            [recommender],
            [0],
            len_trial,
            environment_names=[environment_name],
            recommender_names=[recommender_name[j]],
            bucket_name=bucket_name,
            data_dir=data_dir,
            overwrite=overwrite)