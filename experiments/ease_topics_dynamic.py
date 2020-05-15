import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from run_utils import get_env_dataset, run_env_experiment
from run_utils import ModelTuner
from reclab.environments import Topics
from env_defaults import TOPICS_STATIC, get_len_trial, TOPICS_DYNAMIC
from reclab.recommenders import EASE

# ====Step 4====
# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'master'
overwrite = True

# Experiment setup.
n_trials = 10
trial_seeds = [i for i in range(n_trials)]
len_trial = get_len_trial(TOPICS_DYNAMIC)

# Environment setup
environment_name = TOPICS_DYNAMIC['name']
env = Topics(**TOPICS_DYNAMIC['params'], **TOPICS_DYNAMIC['optional_params'])

# Recommender setup
recommender_name = 'EASE'
recommender_class = EASE

# Tuning is the same as the static case

# Set regularization to 5000.
lam = 5000

# ====Step 7====
recommender = recommender_class(lam=lam)
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
