import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
sys.path.append('experiments')
sys.path.append('.')
sys.path.append('experiments/experiments')
from env_defaults import LATENT_STATIC, get_len_trial
from llorma_optimal_params import OPT_LATENT, LEARNING_RATE, LAMBDA_VAL
from run_utils import get_env_dataset, run_env_experiment
from reclab.environments import LatentFactorBehavior
from reclab.recommenders import Llorma


# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'Mihaela'
overwrite = True


ENV_DICT = LATENT_STATIC
len_trial = get_len_trial(ENV_DICT)
trial_seed = [0]

# Environment setup
environment_name = ENV_DICT['name']
# env = LatentFactorBehavior(**ENV_DICT['params'], **ENV_DICT['optional_params'])
env = LatentFactorBehavior(**ENV_DICT['params'], **ENV_DICT['optional_params'])

i = 0
# Recommender setup
recommender_name = ['Llorma-{}-0'.format(i),'Llorma-{}-1'.format(i),'Llorma-{}-2'.format(i),'Llorma-{}-3'.format(i)]
recommender_class = Llorma

learning_rate = LEARNING_RATE[i]
recommenders = [recommender_class(max_user=ENV_DICT['params']['num_users'],
                                 max_item=ENV_DICT['params']['num_items'],
                                 learning_rate = learning_rate,
                                 lambda_val = LAMBDA_VAL[j],
                                 **OPT_LATENT,
                                ) for j in range(len(LAMBDA_VAL))]

for i, recommender in enumerate(recommenders):
    run_env_experiment(
            [env],
            [recommender],
            trial_seed,
            len_trial,
            environment_names=[environment_name],
            recommender_names=[recommender_name[i]],
            bucket_name=bucket_name,
            data_dir=data_dir,
            overwrite=overwrite)
