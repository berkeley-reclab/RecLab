import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from env_defaults import LATENT_STATIC
from run_utils import get_env_dataset, run_env_experiment
from run_utils import ModelTuner
from reclab.environments import LatentFactorBehavior
from reclab.recommenders import LibFM

assert len(sys.argv) >= 3
strategy = sys.argv[1]
lowdata = sys.argv[2] in ['1', 'True', 'true']

# ====Step 4====
# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'Sarah'
overwrite = True

# Modifying Initial and Final Rating Numbers for Low Data Regime
if lowdata:
  LATENT_STATIC['optional_params']['num_init_ratings'] = 1000
  LATENT_STATIC['misc']['num_final_ratings'] = 101000
  LATENT_STATIC['name'] += '_lowdata'

# Experiment setup.
num_users = LATENT_STATIC['params']['num_users']
num_init_ratings = LATENT_STATIC['optional_params']['num_init_ratings']
num_final_ratings = LATENT_STATIC['misc']['num_final_ratings']
rating_frequency = LATENT_STATIC['optional_params']['rating_frequency']
len_trial = math.ceil((num_final_ratings - num_init_ratings) /
                      (num_users * rating_frequency))

if len(sys.argv) > 3:
    trial_seeds = list(np.fromstring(sys.argv[3], sep=',').astype(int))
else:
    n_trials = 10
    trial_seeds = [i for i in range(n_trials)]

# Environment setup
environment_name = LATENT_STATIC['name']
env = LatentFactorBehavior(**LATENT_STATIC['params'], **LATENT_STATIC['optional_params'])

# Recommender setup
recommender_name = 'LibFM (SGD)'
if strategy != 'greedy':
  recommender_name += ' ' + strategy
recommender_class = LibFM


# ====Step 6====
# Using tuned parameters from topics_static
num_two_way_factors = 20
num_iter = 200
init_stdev = 1.0
learning_rate = 0.0032500000000000003
reg = 0.1
default_params = dict(num_user_features=0,
                      num_item_features=0,
                      num_rating_features=0,
                      max_num_users=num_users,
                      max_num_items=LATENT_STATIC['params']['num_items'],
                      method='sgd')

# ====Step 7====
recommender = recommender_class(num_iter=num_iter,
                                num_two_way_factors=num_two_way_factors,
                                init_stdev=init_stdev,
                                learning_rate=learning_rate,
                                reg=reg,
                                strategy=strategy,
                                **default_params)
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
