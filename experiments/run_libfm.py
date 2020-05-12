import numpy as np
import matplotlib.pyplot as plt
import sys

from run_utils import run_env_experiment, plot_ratings_mses, ModelTuner

sys.path.append('../') 
from reclab.environments import Topics
from reclab.environments import Engelhardt
from reclab.recommenders import LibFM
## Key Parameters across all settings
expdirname = 'dynamic_user_static_rec'

topics = True

num_users = 1000
num_items = 1700
env_params = {
    'rating_frequency': 0.2,
    'num_topics': 19,
    'num_init_ratings': 100000,
    'num_users': 1000,
    'num_items': 1700
}

exp_params = {
    'n_trials': 10,
    'len_trial': 500,
    'SEED': 24532,
}
num_init_ratings = 100000

rec = LibFM(
        num_user_features=0,
        num_item_features=0,
        num_rating_features=0,
        max_num_users=num_users,
        max_num_items=num_items,
        method='sgd',
        learning_rate=0.03,
        bias_reg=0.1,
        one_way_reg=0.1,
        two_way_reg=0.1
    )

env = Topics(**env_params)

ratings, preds, dense_ratings, dense_preds = run_env_experiment([env], [rec], np.arange(1), 500, environment_names=[env.name], recommender_names=['libfm_0.03'], bucket_name='recsys-eval', data_dir='Alex', overwrite=True)
