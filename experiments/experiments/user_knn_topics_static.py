import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from env_defaults import TOPICS_STATIC
from run_utils import get_env_dataset, run_env_experiment
from run_utils import ModelTuner
from reclab.environments import Topics
from reclab.recommenders import KNNRecommender

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
recommender_name = 'UserKnn'
recommender_class = KNNRecommender


# ====Step 5====
starting_data = get_env_dataset(env)


# ====Step 6====
# Recommender tuning setup
n_fold = 5
default_params = dict(user_based=True)
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
print("Larger neighborhood sizes should lead to increased performance.")
shrinkages = [0]
neighborhood_sizes = np.linspace(10, 1001, 10, dtype=np.int).tolist()
tuner.evaluate_grid(neighborhood_size=neighborhood_sizes,
                    shrinkage=shrinkages)

# Set neighborhood size to tradeoff runtime and performance.
neighborhood_size = 250

# Tune the performance independent hyperparameters.
# Start with a coarse grid.
shrinkages = np.linspace(0, 1000, 10).tolist()
neighborhood_sizes = [neighborhood_size]
tuner.evaluate_grid(neighborhood_size=neighborhood_sizes,
                    shrinkage=shrinkages)

# It seems that shrinkage doesn't have much of an effect here but let's
# refine the grid just in case.
shrinkages = np.linspace(0, 10, 10).tolist()
neighborhood_sizes = [neighborhood_size]
tuner.evaluate_grid(neighborhood_size=neighborhood_sizes,
                    shrinkage=shrinkages)

# Set shrinkage to zero since its value doesn't seem to have an effect.
shrinkage = 0

# ====Step 7====
recommender = recommender_class(shrinkage=shrinkage, neighborhood_size=neighborhood_size)
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
