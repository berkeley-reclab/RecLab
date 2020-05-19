import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from env_defaults import TOPICS_STATIC
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
num_users = TOPICS_STATIC['params']['num_users']
num_init_ratings = TOPICS_STATIC['optional_params']['num_init_ratings']
num_final_ratings = TOPICS_STATIC['misc']['num_final_ratings']
rating_frequency = TOPICS_STATIC['optional_params']['rating_frequency']
n_trials = 10
len_trial = math.ceil((num_final_ratings - num_init_ratings) /
                      (num_users * rating_frequency))
trial_seeds = [i for i in range(n_trials)]

# Environment setup
environment_name = TOPICS_STATIC['name']
env = Topics(**TOPICS_STATIC['params'], **TOPICS_STATIC['optional_params'])

# Recommender setup
recommender_name = 'Autorec'
recommender_class = Autorec


# ====Step 5====
starting_data = get_env_dataset(env)


# ====Step 6====
# Recommender tuning setup
n_fold = 5
default_params = dict(num_users=num_users,
                      num_items=TOPICS_STATIC['params']['num_items'])
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
train_epoch = [200]
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
lambda_value = np.linspace(0, 1, 10).tolist()
lrs = np.linspace(1e-4, 1e-2, 10).tolist()
decays = np.linspace(1, 100, 5).tolist()

results = tuner.evaluate_grid(train_epoch=train_epochs,
                    hidden_neuron=hidden_neurons,
                    lambda_value=lambda_value,
                    base_lr=lrs,
                    decay_epoch_step=decays)

# Set parameters based on tuning
best_params = results[results['average_mse'] == results['average_mse'].min()]
lambda_value = float(best_params['lambda_value'])
lr = float(best_params['base_lr'])
decay_epoch_step = float(best_params['decay_epoch_step'])

# ====Step 7====
recommender = recommender_class(train_epoch=train_epoch,
                                hidden_neuron=hidden_neuron,
                                lambda_value=lambda_value,
                                decay_epoch_step=decay_epoch_step,
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
