import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from run_utils import get_env_dataset, run_env_experiment
from run_utils import ModelTuner
from reclab.environments import Topics
from env_defaults import TOPICS_STATIC, TOPICS_DYNAMIC, LATENT_STATIC, LATENT_DYNAMIC, get_len_trial
from reclab.recommenders import PerfectRec

if env_name == 'topics_static':
    ENV_PARAMS = TOPICS_STATIC
    EnvObj = Topics
elif env_name == 'topics_dynamic':
    ENV_PARAMS = TOPICS_DYNAMIC
    EnvObj = Topics
elif env_name == 'latent_static':
    ENV_PARAMS = LATENT_STATIC
    EnvObj = LatentFactorBehavior
elif env_name == 'latent_dynamic':
    ENV_PARAMS = LATENT_DYNAMIC
    EnvObj = LatentFactorBehavior
else:
    assert False, "environment not implemented!"

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
env = EnvObj(**ENV_PARAMS['params'], **ENV_PARAMS['optional_params'])

# Recommender setup
recommender_name = 'PerfectRec'
recommender_class = PerfectRec

def rating_func(user_id, item_id):
    ratings = env.dense_ratings
    return ratings[user_id, item_id]


# ====Step 5====
# Skipping, since nothing to tune

# ====Step 6====
# Skipping, since nothing to tune

# ====Step 7====
recommender = recommender_class(rating_func)
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
