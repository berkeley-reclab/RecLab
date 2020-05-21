import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
sys.path.append('experiments')
sys.path.append('.')
sys.path.append('experiments/experiments')
from env_defaults import LATENT_STATIC_SMALL, get_len_trial
from run_utils import get_env_dataset, run_env_experiment
from run_utils import ModelTuner
from reclab.environments import LatentFactorBehavior, Topics
from reclab.recommenders import Llorma

# ====Step 4====
# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'master'
overwrite = True

# Experiment setup.
#num_users = LATENT_STATIC['params']['num_users']
#num_init_ratings = LATENT_STATIC['optional_params']['num_init_ratings']
#num_final_ratings = LATENT_STATIC['misc']['num_final_ratings']
#rating_frequency = LATENT_STATIC['optional_params']['rating_frequency']
n_trials = 3
ENV_DICT = LATENT_STATIC_SMALL
len_trial = get_len_trial(ENV_DICT)
trial_seeds = [i for i in range(n_trials)]

# Environment setup
environment_name = ENV_DICT['name']
# env = LatentFactorBehavior(**ENV_DICT['params'], **ENV_DICT['optional_params'])
env = LatentFactorBehavior(**ENV_DICT['params'], **ENV_DICT['optional_params'])

# Recommender setup
recommender_name = 'Llorma'
recommender_class = Llorma


# ====Step 5====
starting_data = get_env_dataset(env)


# ====Step 6====
# Recommender tuning setup
n_fold = 5
default_params = dict(max_user=ENV_DICT['params']['num_users'],
                      max_item=ENV_DICT['params']['num_items'],
                      pre_rank=5,
                      rank=10,
                      pre_train_steps=20,
                      train_steps=100,
                      use_cache=True,
                      n_anchor=10,
                      result_path="results_latent2"
                      )

tuner = ModelTuner(starting_data,
                   default_params,
                   recommender_class,
                   n_fold=n_fold,
                   verbose=True,
                   bucket_name=bucket_name,
                   data_dir=data_dir,
                   environment_name=environment_name,
                   recommender_name=recommender_name,
                   overwrite=overwrite)


# do a set of coarse experiments

#n_anchor=[3, 10, 20],
pre_learning_rate=[1e-4, 1e-3, 2e-3]
pre_lambda_val=[0.1, 1, 10]
learning_rate=[1e-4, 1e-3, 1e-2, 1e-1]
lambda_val=[1e-4, 1e-3, 1e-2]
batch_size=[32, 128]

# n_anchor=[3],
# pre_learning_rate=[1e-3]
# pre_lambda_val=[1]
# learning_rate=[1e-4]
# lambda_val=[1e-3]
# batch_size=[128]

coarse_res = tuner.evaluate_grid(pre_learning_rate=pre_learning_rate,
                                pre_lambda_val=pre_lambda_val,
                                learning_rate=learning_rate,
                                lambda_val=lambda_val,
                                batch_size=batch_size)

best_params = coarse_res[coarse_res['average_metric'] == coarse_res['average_metric'].min()]
#n_anchor = int(best_params['n_anchor'])
pre_learning_rate  =  float(best_params['pre_learning_rate'])
pre_lambda_val =  float(best_params['pre_lambda_val'])
learning_rate  =  float(best_params['pre_learning_rate'])
lambda_val =  float(best_params['pre_lambda_val'])
batch_size = int(best_params['batch_size'])

print(best_params)
recommender = recommender_class(**default_params,
                                pre_learning_rate=pre_learning_rate,
                                pre_lambda_val=pre_lambda_val,
                                learning_rate=learning_rate,
                                lambda_val=lambda_val,
                                batch_size=batch_size,
                                )

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
