import numpy as np
import sys, os
sys.path.append('../') 

from reclab.recommenders.libfm.libfm import LibFM
import matplotlib.pyplot as plt

def plot_ratings_mses(mean_ratings, mses, env_params):
    """ Plotting functionality. """
    plt.figure(figsize=[9,4])
    xs = env_params['num_init_ratings'] + env_params['num_users']*env_params['rating_frequency'] * np.arange(mean_ratings.shape[1])
    plt.subplot(1,2,1)
    plt.plot(xs, np.mean(mean_ratings, axis=0), label='mean rating')
    plt.xlabel('# ratings'); plt.ylabel('mean rating')
    plt.subplot(1,2,2)
    plt.plot(xs, np.mean(mses, axis=0), label='mse')
    plt.xlabel('# ratings'); plt.ylabel('mse')
    plt.tight_layout()
    plt.show()

def run_env_experiment(env, exp_params, env_params, params, expdirname, datafilename, overwrite=False):
    """ Main functionality for repeated experiments. """
    datadirname = os.path.join('data', expdirname)
    if not os.path.exists('data/'):
        os.makedirs('data/')
    if not os.path.exists(datadirname):
        os.makedirs(datadirname)
    filename = os.path.join(datadirname, datafilename)

    if not os.path.exists(filename) or overwrite:
        all_mean_ratings = []; all_mses = []
        recommender = LibFM(num_user_features=0, num_item_features=0, 
                    num_rating_features=0, max_num_users=env_params['num_users'], 
                    max_num_items=env_params['num_items'])
        for i in range(exp_params['n_trials']):
            mean_ratings, mses = run_trial(env, recommender, exp_params['len_trial'])
            all_mean_ratings.append(mean_ratings)
            all_mses.append(mses)
        all_mean_ratings = np.array(all_mean_ratings)
        all_mses = np.array(all_mses)
        np.savez(filename, all_mean_ratings=all_mean_ratings, all_mses=all_mses,
                 params=params, env_params=env_params, exp_params=exp_params)
        print('saving to', filename)
    else:
        print('reading from', filename)
        data = np.load(filename, allow_pickle=True)
        all_mean_ratings = data['all_mean_ratings']
        all_mses = data['all_mses']
        if data['params'] != params: 
            print('Warning: params differ.')
        if data['env_params'] != env_params: 
            print('Warning: env_params differ.')
        if data['exp_params'] != exp_params: 
            print('Warning: exp_params differ.')
    return all_mean_ratings, all_mses 

def run_trial(env, recommender, len_trial):
    """ Logic for running each trial. """

    # First generate the items and users to seed the dataset.
    print("Initializing environment and recommender")
    items, users, ratings = env.reset()
    recommender.reset(items, users, ratings)

    mean_ratings = []
    mses = []
    # Now recommend items to users.
    print("Making online recommendations")
    for i in range(len_trial):
        online_users = env.online_users()
        ret, predicted_ratings = recommender.recommend(online_users, num_recommendations=1)
        recommendations = ret[:, 0]
        items, users, ratings, info = env.step(recommendations)
        recommender.update(users, items, ratings)
        rating_arr = []
        for (rating, _), pred in zip(ratings.values(), predicted_ratings):
            rating_arr.append([rating, pred])
        rating_arr = np.array(rating_arr)
        errors = rating_arr[:,0] - rating_arr[:,1]
        mean_ratings.append(np.mean(rating_arr[:, 0]))
        mses.append(np.mean(errors**2))
        print("Iter:", i, "Mean:", mean_ratings[-1], "MSE:", mses[-1])

    ratings = env.all_ratings()
    return mean_ratings, mses