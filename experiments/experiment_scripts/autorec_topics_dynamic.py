import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from env_defaults import *
from run_utils import get_env_dataset, run_env_experiment
from run_utils import ModelTuner
from reclab.environments import Topics
from reclab.recommenders import Autorec

# ====Step 4====
# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'master'
overwrite = True

# Experiment setup.
n_trials = 10
len_trial = get_len_trial(TOPICS_DYNAMIC)
trial_seeds = [i for i in range(n_trials)]

# Environment setup
environment_name = TOPICS_DYNAMIC['name']
env = Topics(**TOPICS_DYNAMIC['params'], **TOPICS_DYNAMIC['optional_params'])

# Recommender setup
recommender_name = 'Autorec'
recommender_class = Autorec


# ====Step 5====
starting_data = get_env_dataset(env)


# ====Step 6====
# Recommender tuning setup
n_fold = 5
num_users, num_items = get_num_users_items(TOPICS_DYNAMIC)
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

# Verify that the performance dependent hyperparameters lead to increased performance.
print("More hidden neurons should lead to increased performance.")
train_epoch = [1000]
hidden_neuron = np.linspace(1, 1000, 10, dtype=np.int).tolist()
tuner.evaluate_grid(hidden_neuron=hidden_neuron, train_epoch=train_epoch)

# Hidden neuron set based on Autorec paper
hidden_neuron = 500

print("More iterations should lead to increased performance.")
train_epoch = np.linspace(1, 1000, 10, dtype=np.int).tolist()
hidden_neurons = [hidden_neuron]
tuner.evaluate_grid(hidden_neuron=hidden_neurons,
                    train_epoch=train_epoch)

# Set number of iterations to tradeoff runtime and performance.
train_epoch = 200

# Tune the performance independent hyperparameters.
train_epochs = [train_epoch]
hidden_neurons = [hidden_neuron]
lambda_values = np.linspace(0, 10, 5).tolist()
lrs = np.linspace(1e-4, 1e-2, 5).tolist()

results = tuner.evaluate_grid(train_epoch=train_epochs,
                    hidden_neuron=hidden_neurons,
                    lambda_value=lambda_values,
                    base_lr=lrs)

# Set parameters based on tuning
best_params = results[results['average_metric'] == results['average_metric'].min()]
lambda_value = float(best_params['lambda_value'])
lr = float(best_params['base_lr'])

# ====Step 7====
recommender = recommender_class(train_epoch=1000,
                                hidden_neuron=hidden_neuron,
                                lambda_value=lambda_value,
                                base_lr=lr,
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
