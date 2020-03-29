import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from reclab.recommenders import LibFM


def plot_ratings_mses(ratings,
                      predictions,
                      num_init_ratings,
                      labels):
    """Plotting functionality."""
    def get_stats(arr):
        # Swap the trial and step axes.
        arr = np.swapaxes(arr, 0, 1)
        # Flatten the trial and user axes together.
        arr = arr.reshape(arr.shape[0], -1)
        # Compute the means and standard deviations of the means for each step.
        means = arr.mean(axis=1)
        # Use Bessel's correction here.
        stds = arr.std(axis=1) / np.sqrt(arr.shape[1] - 1)
        # Compute the 95% confidence intervals using the CLT.
        upper_bounds = means + 2 * stds
        lower_bounds = np.maximum(means - 2 * stds, 0)
        return means, lower_bounds, upper_bounds

    plt.figure(figsize=[9, 4])
    xs = num_init_ratings + ratings.shape[3] * np.arange(ratings.shape[2])
    plt.subplot(1, 2, 1)
    for recommender_ratings, label in zip(ratings, labels):
        means, lower_bounds, upper_bounds = get_stats(recommender_ratings)
        plt.plot(xs, means, label=label)
        plt.fill_between(xs, lower_bounds, upper_bound, alpha=0.1)
    plt.xlabel('# ratings')
    plt.ylabel('Mean Rating')
    plt.legend()

    plt.subplot(1, 2, 2)
    squared_diffs = (ratings - predictions) ** 2
    for recommender_squared_diffs, label in zip(squared_diffs, labels):
        mse, lower_bounds, upper_bounds = get_stats(squared_diffs)
        # Transform the MSE into the RMSE and correct the associated intervals.
        rmse = np.sqrt(mse)
        lower_bounds = np.sqrt(lower_bounds)
        upper_bounds = np.sqrt(upper_bounds)
        plt.plot(xs, rmse, label=label)
        plt.fill_betweem(xs, lower_bounds, upper_bounds, alpha=0.1)
    plt.xlabel('# ratings')
    plt.ylabel('RMSE')
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_env_experiment(env,
                       recommenders,
                       len_trial,
                       n_trials,
                       exp_dirname,
                       data_filename,
                       overwrite=False):
    """Run repeated trials for a given list of recommenders on a fixed environment.

    Parameters
    ----------
    env : Environment
        The environment to run the experiments with.
    recommenders : list of Recommender
        The recommenders to run the experiments with.
    len_trial : int
        The number of steps to run each trial for.
    n_trials : int
        The number of trials to run for each recommender.
    exp_dirname : str
        The directory in which to save or load the experiment results.
    data_filename : str
        The name of the file in which to save or load the experiment results.
    overwrite : bool
        Whether to re-run the experiment even if a matching file is found.

    Returns
    -------
    ratings : np.ndarray
        The array of all ratings made by users throughout all trials. ratings[i, j, k, l]
        corresponds to the rating made by the l-th online user during the k-th step of the
        j-th trial for the i-th recommender.
    predictions : np.ndarray
        The array of all predictions made by recommenders throughout all trials.
        predictions[i, j, k, l] corresponds to the prediction that the i-th recommender
        made on the rating of the l-th online user during the k-th step of the
        j-th trial for the aforementioned recommender. Note that if the recommender does
        not make predictions to make recommendations then that element will be np.nan.

    """
    datadirname = os.path.join('data', expdirname)
    os.makedirs(datadirname, exist_ok=True)

    filename = os.path.join(datadirname, datafilename)
    if not os.path.exists(filename) or overwrite:
        all_ratings = []
        all_predictions = []
        for recommender in recommenders:
            all_ratings.append([])
            all_predictions.append([])
            for i in range(n_trials):
                ratings, predictions = run_trial(env, recommender, len_trial)
                all_ratings[-1].append(ratings)
                all_predictions[-1].append(predictions)
        all_ratings = np.array(all_ratings)
        all_predictions = np.array(all_predictions)
        np.savez(filename, all_ratings=all_ratings, all_predictions=all_predictions)
        print('Saving to', filename)
    else:
        print('Reading from', filename)
        data = np.load(filename, allow_pickle=True)
        all_ratings = data['all_ratings']
        all_predictions = data['all_predictions']

    return all_ratings, all_predictions


def run_trial(env, recommender, len_trial):
    """Logic for running each trial.

    Parameters
    ----------
    env : Environment
        The environment to use for this trial.
    recommender : Recommender
        The recommender to use for this trial.
    len_trial : int
        The number of recommendation steps to run the trial for.

    Returns
    -------
    ratings : np.ndarray
        The array of all ratings made by users. ratings[i, j] is the rating
        made on round i by the j-th online user on the item recommended to them.
    predictions : np.ndarray
        The array of all predictions made by the recommender. preds[i, j] is the
        prediction the user made on round i for the item recommended to the j-th
        user. If the recommender does not predict items then each element is set
        to np.nan.

    """
    # First generate the items and users to seed the dataset.
    items, users, ratings = env.reset()
    recommender.reset(items, users, ratings)

    all_ratings = []
    all_predictions = []
    # Now recommend items to users.
    for i in range(len_trial):
        online_users = env.online_users()
        recommendations, predictions = recommender.recommend(online_users, num_recommendations=1)
        recommendations = recommendations[:, 0]
        items, users, ratings, info = env.step(recommendations)
        recommender.update(users, items, ratings)
        ratings = [rating, _ in ratings]
        all_ratings.append(ratings)
        if predictions is None:
            predictions = np.ones(ratings.shape) * np.nan
        all_predictions.append(predictions)

    return all_ratings, all_predictions
