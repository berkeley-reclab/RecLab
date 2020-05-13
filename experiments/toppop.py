import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from env_defaults import TOPICS_STATIC, TOPICS_DYNAMIC, LATENT_STATIC, LATENT_DYNAMIC, get_len_trial
from run_utils import run_env_experiment
from reclab.environments import Topics
from reclab.recommenders import TopPop

env_name = str(sys.argv[1])
if env_name == 'topics_static':
    ENV_PARAMS = TOPICS_STATIC
elif env_name == 'topics_dynamic':
    ENV_PARAMS = TOPICS_DYNAMIC
elif env_name == 'latent_static':
    ENV_PARAMS = LATENT_STATIC
elif env_name == 'latent_dynamic':
    ENV_PARAMS = LATENT_DYNAMIC
else:
    assert False, 'environment not implemented!'

# ====Step 4====
# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'master'
overwrite = True

# Experiment setup.
n_trials = 10
trial_seeds = [i for i in range(n_trials)]
len_trial = get_len_trial(ENV_PARAMS)

# Environment setup
environment_name = ENV_PARAMS['name']
env = Topics(**ENV_PARAMS['params'], **ENV_PARAMS['optional_params'])

# Recommender setup
recommender_name = 'TopPop'
recommender_class = TopPop

# ====Step 5====
# Skipping, since nothing to tune

# ====Step 6====
# Skipping, since nothing to tune

# ====Step 7====
recommender = recommender_class()
for i, seed in enumerate(trial_seeds):
    run_env_experiment([env],
                       [recommender],
                       [seed],
                       len_trial,
                       environment_names=[environment_name],
                       recommender_names=[recommender_name],
                       bucket_name=bucket_name,
                       data_dir=data_dir,
                       overwrite=overwrite)