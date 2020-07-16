import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from run_utils import get_env_dataset, run_env_experiment
from run_utils import ModelTuner
from reclab.environments import Topics, LatentFactorBehavior
from env_defaults import LATENT_DYNAMIC, get_len_trial
from reclab.recommenders.cfnade import Cfnade

# ====Step 4====
# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'master'
overwrite = True


# Experiment setup.
num_users = LATENT_DYNAMIC['params']['num_users']
num_init_ratings = LATENT_DYNAMIC['optional_params']['num_init_ratings']
num_final_ratings = LATENT_DYNAMIC['misc']['num_final_ratings']
rating_frequency = LATENT_DYNAMIC['optional_params']['rating_frequency']
n_trials = 10
len_trial = get_len_trial(LATENT_DYNAMIC)
trial_seeds = [i for i in range(n_trials)]

# Environment setup
environment_name = LATENT_DYNAMIC['name']
env = LatentFactorBehavior(**LATENT_DYNAMIC['params'], **LATENT_DYNAMIC['optional_params'])

# Recommender setup
recommender_name = 'CFNade'
recommender_class = Cfnade

# Tuning is the same as the static case

# Set parameters based on tuning

train_epoch = 30
batch_size = 128
hidden_dim = 500
learning_rate = 0.001

# ====Step 7====
recommender = recommender_class(num_users=num_users,
                                num_items=LATENT_DYNAMIC['params']['num_items'], 
                                batch_size=batch_size,
                                train_epoch=train_epoch,
                                hidden_dim=hidden_dim, 
                                learning_rate=learning_rate)

trial_seeds = [0]

#trial_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]


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
