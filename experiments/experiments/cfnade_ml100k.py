import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from run_utils import get_env_dataset, run_env_experiment
from run_utils import ModelTuner
from run_utils import plot_ratings_mses_s3
from env_defaults import ML_100K, get_len_trial
from reclab.recommenders.cfnade import Cfnade
from reclab.environments import DatasetLatentFactor

# ====Step 4====
# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'master'
overwrite = True

# Experiment setup.
num_users = ML_100K['misc']['dataset_size'][0]
num_items = ML_100K['misc']['dataset_size'][1]
num_init_ratings = ML_100K['optional_params']['num_init_ratings']
num_final_ratings = ML_100K['misc']['num_final_ratings']
rating_frequency = ML_100K['optional_params']['rating_frequency']
n_trials = 10
len_trial = get_len_trial(ML_100K)
trial_seeds = [i for i in range(n_trials)]

# Environment setup
environment_name = ML_100K['name']
env = DatasetLatentFactor(**ML_100K['params'], **ML_100K['optional_params'])

# Recommender setup
recommender_name = 'CFNade'
recommender_class = Cfnade


# ====Step 5====
starting_data = get_env_dataset(env)


#====Step 6====
#Recommender tuning setup
n_fold = 5

default_params = dict(num_users=num_users,
                      num_items=num_items)

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

# #Verify that the performance dependent hyperparameters lead to increased performance.
# print("More train epochs should lead to increased performance.")
# train_epochs = [10, 20, 30] #RMSE results:0.665; 0.618; 0.613
# hidden_dims = [500]
# learning_rates =[0.001]
# batch_sizes = [64]
# tuner.evaluate_grid(train_epoch=train_epochs,
#                     hidden_dim=hidden_dims,
#                     learning_rate = learning_rates,
#                     batch_size = batch_sizes)


# print("Smaller batch size should lead to increased performance.")
# train_epochs = [10]
# hidden_dims = [500]
# learning_rates =[0.001]
# batch_sizes = [64, 128, 256, 512] #RMSE results: 0.655; 0.689; 0.795; 0.780
# tuner.evaluate_grid(train_epoch=train_epochs,
#                     hidden_dim=hidden_dims,
#                     learning_rate = learning_rates,
#                     batch_size = batch_sizes)

# print("Larger hidden dim should lead to increased performance.")

# train_epochs = [30]
# hidden_dims = [100, 250, 500] #RMSE results: 0.621; 0.610; 0.617
# #RMSE results: 0.792; 0.795; 0.796 (bs=256, epoch=10)
# learning_rates =[0.001]
# batch_sizes = [64]
# tuner.evaluate_grid(train_epoch=train_epochs,
#                     hidden_dim=hidden_dims,
#                     learning_rate = learning_rates,
#                     batch_size = batch_sizes)

train_epoch = 30
batch_size = 64
hidden_dim = 250

# Tune the performance independent hyperparameters.

# learning_rates = [0.0001, 0.001, 0.01] #RMSE results: 0.790; 0.652; nan
# train_epochs = [train_epoch]
# hidden_dims = [hidden_dim]
# batch_sizes = [batch_size]

# tuner.evaluate_grid(train_epoch=train_epochs,
#                     hidden_dim=hidden_dims,
#                     learning_rate = learning_rates,
#                     batch_size = batch_sizes)

# Set parameters based on tuning
learning_rate = 0.001

# ====Step 7====
recommender = recommender_class(num_users=num_users,
                                 num_items=num_items, 
                                 batch_size=batch_size,
                                 train_epoch=train_epoch,
                                 hidden_dim=hidden_dim, 
                                 learning_rate=learning_rate)

trial_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
