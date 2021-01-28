import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from llorma_optimal_params import OPT_TOPICS
from env_defaults import TOPICS_STATIC
from experiment import get_env_dataset, run_env_experiment
from tuner import ModelTuner
from reclab.environments import Topics
from reclab.recommenders import LibFM
from reclab.recommenders import Llorma

assert len(sys.argv) >= 3
strategy = sys.argv[1]
lowdata = sys.argv[2] in ['1', 'True', 'true']

# ====Step 4====
# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'lowdata'
overwrite = True

# Modifying Initial and Final Rating Numbers for Low Data Regime
if lowdata:
  TOPICS_STATIC['optional_params']['num_init_ratings'] = 1000
  TOPICS_STATIC['misc']['num_final_ratings'] = 101000
  TOPICS_STATIC['name'] += '_lowdata'

# Experiment setup.
num_users = TOPICS_STATIC['params']['num_users']
num_init_ratings = TOPICS_STATIC['optional_params']['num_init_ratings']
num_final_ratings = TOPICS_STATIC['misc']['num_final_ratings']
rating_frequency = TOPICS_STATIC['optional_params']['rating_frequency']
len_trial = math.ceil((num_final_ratings - num_init_ratings) /
                      (num_users * rating_frequency))

if len(sys.argv) > 3:
    trial_seeds = list(np.fromstring(sys.argv[3], sep=',').astype(int))
else:
    n_trials = 10
    trial_seeds = [i for i in range(n_trials)]

# Environment setup
environment_name = TOPICS_STATIC['name']
env = Topics(**TOPICS_STATIC['params'], **TOPICS_STATIC['optional_params'])

# Recommender setup
recommender_name = 'Llorma'
if strategy != 'greedy':
  recommender_name += ' ' + strategy
recommender_class = Llorma

recommender = recommender_class(max_user=TOPICS_STATIC['params']['num_users'],
                                max_item=TOPICS_STATIC['params']['num_items'],
                                **OPT_TOPICS)

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
