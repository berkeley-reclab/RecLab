import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from env_defaults import ENGELHARDT
from run_utils import get_env_dataset, run_env_experiment
from run_utils import ModelTuner
from reclab.environments import Engelhardt
from reclab.recommenders import Autorec

# ====Step 4====
# S3 storage parameters
bucket_name = 'recsys-eval'
data_dir = 'master'
overwrite = True

# Experiment setup.
num_users = ENGELHARDT['params']['num_users']
num_init_ratings = ENGELHARDT['optional_params']['num_init_ratings']
num_final_ratings = ENGELHARDT['misc']['num_final_ratings']
rating_frequency = ENGELHARDT['optional_params']['rating_frequency']
n_trials = 10
len_trial = math.ceil((num_final_ratings - num_init_ratings) /
                      (num_users * rating_frequency))
trial_seeds = [i for i in range(n_trials)]

# Environment setup
environment_name = ENGELHARDT['name']
env = Engelhardt(**ENGELHARDT['params'], **ENGELHARDT['optional_params'])

# Recommender setup
recommender_name = 'Autorec'
recommender_class = Autorec


# ====Step 5====
starting_data = get_env_dataset(env)


# ====Step 6====
# Recommender tuning setup
n_fold = 5
default_params = dict(num_users=num_users,
                      num_items=ENGELHARDT['params']['num_items'])
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
train_epoch = 1000

# Tune the performance independent hyperparameters.
train_epochs = [train_epoch]
hidden_neurons = [hidden_neuron]
lambda_values = np.linspace(0, 10, 10).tolist()
lrs = np.linspace(1e-4, 1e-2, 10).tolist()
dropouts = np.linspace(0, 0.1, 5).tolist()

results = tuner.evaluate_grid(train_epoch=train_epochs,
                    hidden_neuron=hidden_neurons,
                    lambda_value=lambda_values,
                    base_lr=lrs,
                    dropout=dropouts)

# Set parameters based on tuning
best_params = results[results['average_mse'] == results['average_mse'].min()]
lambda_value = float(best_params['lambda_value'])
lr = float(best_params['base_lr'])
dropout = float(best_params['dropout'])

# ====Step 7====
recommender = recommender_class(train_epoch=train_epoch,
                                hidden_neuron=hidden_neuron,
                                lambda_value=lambda_value,
                                dropout=dropout,
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
