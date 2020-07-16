import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from run_utils import get_env_dataset, run_env_experiment
from run_utils import ModelTuner
from reclab.environments import Topics, LatentFactorBehavior, DatasetLatentFactor
from env_defaults import *
from reclab.recommenders import PerfectRec

env_name = str(sys.argv[1])
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
elif env_name == 'ml_100k':
    ENV_PARAMS = ML_100K
    EnvObj = DatasetLatentFactor
elif env_name == 'ml_100k_lowdata':
    ENV_PARAMS = ML_100K_LOWDATA
    EnvObj = DatasetLatentFactor
else:
    assert False, "environment not implemented!"

# ====Step 4====
# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'master'
overwrite = True

# Experiment setup.
if len(sys.argv) > 2:
    trial_seeds = list(np.fromstring(sys.argv[2], sep=',').astype(int))
else:
    n_trials = 10
    trial_seeds = [i for i in range(n_trials)]
len_trial = get_len_trial(ENV_PARAMS)

# Environment setup
environment_name = ENV_PARAMS['name']
env = EnvObj(**ENV_PARAMS['params'], **ENV_PARAMS['optional_params'])

# Recommender setup
recommender_name = 'PerfectRec'
recommender_class = PerfectRec

def rating_func():
    ratings = env.dense_ratings
    return ratings


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
