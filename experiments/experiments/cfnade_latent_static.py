import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from run_utils import get_env_dataset, run_env_experiment
from run_utils import ModelTuner
from run_utils import plot_ratings_mses_s3
from reclab.environments import Topics, LatentFactorBehavior
from env_defaults import LATENT_STATIC, get_len_trial
from reclab.recommenders.cfnade import Cfnade

# ====Step 4====
# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'master'
overwrite = True

# Experiment setup.
num_users = LATENT_STATIC['params']['num_users']
num_init_ratings = LATENT_STATIC['optional_params']['num_init_ratings']
num_final_ratings = LATENT_STATIC['misc']['num_final_ratings']
rating_frequency = LATENT_STATIC['optional_params']['rating_frequency']
n_trials = 10
len_trial = get_len_trial(LATENT_STATIC)
trial_seeds = [i for i in range(n_trials)]

# Environment setup
environment_name = LATENT_STATIC['name']
env = LatentFactorBehavior(**LATENT_STATIC['params'], **LATENT_STATIC['optional_params'])

# Recommender setup
recommender_name = 'CFNade'
recommender_class = Cfnade


# ====Step 5====
starting_data = get_env_dataset(env)


# ====Step 6====
# Recommender tuning setup
n_fold = 5

default_params = dict(num_users=num_users,
                      num_items=LATENT_STATIC['params']['num_items'])

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

#Tuning

# # Verify that the performance dependent hyperparameters lead to increased performance.
# print("More train epochs should lead to increased performance.")
# train_epochs = [10, 20, 30] #results: 0.960; 0.924; 0.890
# hidden_dims = [500]
# learning_rates =[0.001]
# batch_sizes = [512]
# tuner.evaluate_grid(train_epoch=train_epochs,
#                     hidden_dim=hidden_dims,
#                     learning_rate = learning_rates,
#                     batch_size = batch_sizes)


# print("Smaller batch size should lead to increased performance.")
# train_epochs = [30]
# hidden_dims = [500]
# learning_rates =[0.001]
# batch_sizes = [64, 128, 256] #results: 0.789; 0.782; 0.794
# tuner.evaluate_grid(train_epoch=train_epochs,
#                     hidden_dim=hidden_dims,
#                     learning_rate = learning_rates,
#                     batch_size = batch_sizes)

# print("Larger hidden dim should lead to increased performance.")
# train_epochs = [30]
# hidden_dims = [100, 250, 500] #results: 0.897; 0.822; 0.792
# learning_rates =[0.001]
# batch_sizes = [256]
# tuner.evaluate_grid(train_epoch=train_epochs,
#                     hidden_dim=hidden_dims,
#                     learning_rate = learning_rates,
#                     batch_size = batch_sizes)

# # Tune the performance independent hyperparameters.

# learning_rates = [0.0001, 0.001, 0.01] #results: 0.92; 0.844; nan
# train_epochs = [train_epoch]
# hidden_dims = [hidden_dim]
# batch_sizes = [batch_size]

# tuner.evaluate_grid(train_epoch=train_epochs,
#                     hidden_dim=hidden_dims,
#                     learning_rate = learning_rates,
#                     batch_size = batch_sizes)

# Set parameters based on tuning

train_epoch = 30
batch_size = 128
hidden_dim = 500
learning_rate = 0.001

# ====Step 7====
recommender = recommender_class(num_users=num_users,
                                num_items=LATENT_STATIC['params']['num_items'], 
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
