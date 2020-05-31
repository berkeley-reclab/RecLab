"""A utility module for running experiments."""
import codecs
import collections
import copy
import datetime
import io
import json
import os
import pickle
import subprocess

import boto3
import functional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm.autonotebook


sns.set(style="ticks")
plt.rc('font', family='serif')
plt.rc('font', serif='Times New Roman')

bigfont = 18
medfont = 14
smallfont = 12
sns.set_style('ticks', {'font.family':'serif', 'font.serif':'Times New Roman'})

# The random seed that defines the initial state of each environment.
INIT_SEED = 0
# The name of the file temporarily created for uploads to S3.
TEMP_FILE_NAME = 'temp.out'
# The id of all users that will have access to the S3 objects.
AWS_IDS = ['acde32a12806f031eb2518b0c2aca259ba031314143dfe2fab1bf6207af665f0',
           '4367fa3f02d247984d07c4e475f3e4e2abe5f95c6549f494f6784027cbb62601',
           '9fa447cf916c8ee7d02047335fdf9055c1627518e78b7168c6c737a403c3035f',
           '634b7a0686be3590c1808efc465ea9db660233386f1ad0bbbe3cabab19ae2564']
ID_STR = ','.join(['id=' + aws_id for aws_id in AWS_IDS])


def plot_ratings_mses(ratings,
                      predictions,
                      labels,
                      use_median=False,
                      num_init_ratings=None,
                      threshold=10,
                      title=['', '']):
    """Plot the performance results for multiple recommenders.
    Parameters
    ----------
    ratings : np.ndarray
        The array of all ratings made by users throughout all trials. ratings[i, j, k, l]
        corresponds to the rating made by the l-th online user during the k-th step of the
        j-th trial for the i-th recommender.
    predictions : np.ndarray
        The array of all predictions made by recommenders throughout all trials.
        predictions[i, j, k, l] corresponds to the prediction that the i-th recommender
        made on the rating of the l-th online user during the k-th step of the
        j-th trial for the aforementioned recommender. If a recommender does not make
        predictions then its associated elements can be np.nan. It will not be displayed
        during RMSE plotting.
    labels : list of str
        The name of each recommender.
    use_median : bool
        Which type of summary statistics: either False for mean and std deviation
        or True for median and quartiles.
    num_init_ratings : int
        The number of ratings initially available to recommenders. If set to None
        the function will plot with an x-axis based on the timestep.
    threshold: float
        The threshold filtering on the predictions, predictions larger than it will be set to 0.
        default is 10
    """
    if num_init_ratings is not None:
        x_vals = num_init_ratings + ratings.shape[3] * np.arange(ratings.shape[2])
    else:
        x_vals = np.arange(ratings.shape[2])

    # Setting the predictions for a user/item that has no ratings in the training data to 0.
    predictions[predictions > threshold] = 0
    rgblist = sns.color_palette(n_colors=len(labels))

    plt.figure(figsize=[9, 4])
    plt.subplot(1, 2, 1)
    for recommender_ratings, label, rgb in zip(ratings, labels, rgblist):
        means, lower_bounds, upper_bounds = compute_stats(recommender_ratings,
                                                          bound_zero=True,
                                                          use_median=use_median)
        plt.plot(x_vals, means, label=label, color=rgb)
        plt.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.25,
                         color=rgb)
    plt.xlabel('# ratings')
    plt.ylabel('Mean Rating')
    plt.title(title[0])
    plt.legend()

    plt.subplot(1, 2, 2)
    squared_diffs = (ratings - predictions) ** 2
    for recommender_squared_diffs, label, rgb in zip(squared_diffs, labels, rgblist):
        mse, lower_bounds, upper_bounds = compute_stats(recommender_squared_diffs,
                                                        bound_zero=True,
                                                        use_median=use_median)
        # Transform the MSE into the RMSE and correct the associated intervals.
        rmse = np.sqrt(mse)
        lower_bounds = np.sqrt(lower_bounds)
        upper_bounds = np.sqrt(upper_bounds)
        plt.plot(x_vals, rmse, label=label, color=rgb)
        plt.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.25, color=rgb)
    plt.xlabel('# ratings')
    plt.ylabel('RMSE')
    plt.title(title[1])
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ratings_mses_s3(rec_names,
                         labels,
                         len_trial,
                         bucket_name,
                         data_dir_name,
                         env_name,
                         seeds,
                         plot_dense=False,
                         num_users=None,
                         rating_frequency=None,
                         num_init_ratings=None,
                         threshold=10,
                         title=['', '']):
    """Plot the performance results for multiple recommenders using data stored in S3.
    Parameters
    ----------
    labels : list of str
        The name of each recommender.
    len_trial : int
        The length of each trial.
    bucket_name : str
        The bucket in which the experiment data is saved.
    data_dir_name : str
        The name of the directory in which the experiment data is saved.
    env_name : str
        The name of the environment for which we are plotting the results.
    seeds : list of int
        The trial seeds across which we are averaging results.
    plot_dense : bool
        Whether to plot performance numbers using dense ratings and predictions.
    num_users : int
        The number of users. If set to None the function will plot with an x-axis
        based on the timestep
    num_init_ratings : int
        The number of ratings initially available to recommenders. If set to None
        the function will plot with an x-axis based on the timestep.
    threshold: float
        The threshold filtering on the predictions, predictions larger than it will be set to 0.
        default is 10
    """
    rgblist = sns.color_palette('muted', n_colors=len(labels))
    bucket = boto3.resource('s3').Bucket(bucket_name)  # pylint: disable=no-member

    def arr_func(ratings, predictions):
        # Setting the predictions for a user/item that has no ratings in the training data to 0.
        predictions[0][predictions[0] > threshold] = 0
        return [ratings[0], (ratings[0] - predictions[0]) ** 2]

    if num_init_ratings is not None and num_users is not None and rating_frequency is not None:
        x_vals = num_init_ratings + num_users * rating_frequency * np.arange(len_trial)
    else:
        x_vals = np.arange(len_trial)

    all_stats = {}
    for label, rec_name in zip(labels, rec_names):
        all_stats[label] = compute_stats_s3(bucket=bucket,
                                            data_dir_name=data_dir_name,
                                            env_name=env_name,
                                            rec_names=[rec_name],
                                            seeds=seeds,
                                            bound_zero=True,
                                            arr_func=arr_func,
                                            load_dense=plot_dense)

    plt.figure(figsize=[16, 6])
    plt.subplot(1, 2, 1)
    for label, rgb in zip(labels, rgblist):
        means, lower_bounds, upper_bounds = all_stats[label]
        means = means[0]
        lower_bounds = lower_bounds[0]
        upper_bounds = upper_bounds[0]
        plt.plot(x_vals, means, label=label, color=rgb)
        plt.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.25, color=rgb)
    plt.xticks(fontsize=bigfont)
    plt.yticks(fontsize=bigfont)
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    plt.xlabel('Number of Ratings', fontsize=bigfont)
    plt.ylabel('Mean Rating', fontsize=bigfont)
    plt.ylim(bottom=2.75)
    plt.title(title[0])
    plt.legend(fontsize=smallfont, markerscale=1, framealpha=1, frameon=False)

    plt.subplot(1, 2, 2)
    for label, rgb in zip(labels, rgblist):
        means, lower_bounds, upper_bounds = all_stats[label]
        mse = means[1]
        lower_bounds = lower_bounds[1]
        upper_bounds = upper_bounds[1]
        # Transform the MSE into the RMSE and correct the associated intervals.
        rmse = np.sqrt(mse)
        lower_bounds = np.sqrt(lower_bounds)
        upper_bounds = np.sqrt(upper_bounds)
        plt.plot(x_vals, rmse, label=label, color=rgb)
        plt.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.25, color=rgb)
    plt.xticks(fontsize=bigfont)
    plt.yticks(fontsize=bigfont)
    plt.ylim(top=1.8)
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    plt.xlabel('Number of Ratings', fontsize=bigfont)
    plt.ylabel('RMSE', fontsize=bigfont)
    plt.title(title[1])
    plt.legend(fontsize=smallfont, markerscale=1, framealpha=1, frameon=False)
    return all_stats


def plot_regret(ratings,
                labels,
                perfect_ratings=None,
                num_init_ratings=None):
    """Plot the regrets for multiple recommenders comparing to the perfect recommender.
    Parameters
    ----------
    ratings : np.ndarray
        The array of all ratings made by users throughout all trials. ratings[i, j, k, l]
        corresponds to the rating made by the l-th online user during the k-th step of the
        j-th trial for the i-th recommender.
    labels : list of str
        The name of each recommender. Default label for the perfect recommender is 'perfect'.
    perfect_ratings : np.ndarray, can be none if labels contains 'perfect'
        The array of all ratings made for the perfect recommenders thoughout all trials.
        ratings[j, k, l] corresponds to the rating made by the l-th online user during
        the k-th step of the j-th trial for the perfect recommender.
    num_init_ratings : int
        The number of ratings initially available to recommenders. If set to None
        the function will plot with an x-axis based on round number.
    """
    if perfect_ratings is None:
        if 'perfect' in labels:
            idx = labels.index('perfect')
            perfect_ratings = ratings[idx]
        else:
            print('No ratings from the perfect recommender.')
            return

    if num_init_ratings is not None:
        x_vals = num_init_ratings + ratings.shape[3] * np.arange(ratings.shape[2])
    else:
        x_vals = np.arange(ratings.shape[2])

    plt.figure(figsize=[5, 4])
    for recommender_ratings, label in zip(ratings, labels):
        # Plot the regret for the recommenders that are not perfect.
        if label != 'perfect':
            regrets = perfect_ratings - recommender_ratings
            regrets = np.cumsum(regrets, axis=1)
            mean_regrets, lower_bounds, upper_bounds = compute_stats(regrets)
            # Plotting the regret over steps and correct the associated intervals.
            plt.plot(x_vals, mean_regrets, label=label)
            plt.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.1)
    plt.xlabel('# ratings')
    plt.ylabel('Regret')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_regret_s3(labels,
                   len_trial,
                   bucket_name,
                   data_dir_name,
                   env_name,
                   seeds,
                   perfect_name='PerfectRec',
                   plot_dense=False,
                   num_users=None,
                   num_init_ratings=None):
    """Plot the regret for multiple recommenders using data stored in S3.
    Parameters
    ----------
    labels : list of str
        The name of each recommender.
    len_trial : int
        The length of each trial.
    bucket_name : str
        The bucket in which the experiment data is saved.
    data_dir_name : str
        The name of the directory in which the experiment data is saved.
    env_name : str
        The name of the environment for which we are plotting the results.
    seeds : list of int
        The trial seeds across which we are averaging results.
    perfect_name : str
        The name of the recommender against which to compute regret.
    plot_dense : bool
        Whether to plot regret using the dense ratings.
    num_users : int
        The number of users. If set to None the function will plot with an x-axis
        based on the timestep
    num_init_ratings : int
        The number of ratings initially available to recommenders. If set to None
        the function will plot with an x-axis based on the timestep.
    """
    bucket = boto3.resource('s3').Bucket(bucket_name)  # pylint: disable=no-member
    def regret(ratings, predictions):
        return [np.cumsum(ratings[0] - ratings[1], axis=0)]

    if num_init_ratings is not None and num_users is not None:
        x_vals = num_init_ratings + num_users * np.arange(len_trial)
    else:
        x_vals = np.arange(len_trial)

    plt.figure(figsize=[5, 4])
    for label in labels:
        mean_regrets, lower_bounds, upper_bounds = compute_stats_s3(bucket=bucket,
                                                                    data_dir_name=data_dir_name,
                                                                    env_name=env_name,
                                                                    rec_names=[perfect_name, label],
                                                                    seeds=seeds,
                                                                    arr_func=regret,
                                                                    bound_zero=False,
                                                                    load_dense=plot_dense)
        # Plotting the regret over steps and correct the associated intervals.
        plt.plot(x_vals, mean_regrets[0], label=label)
        plt.fill_between(x_vals, lower_bounds[0], upper_bounds[0], alpha=0.1)
    plt.xlabel('# ratings')
    plt.ylabel('Regret')
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_stats(arr, bound_zero=False, use_median=False):
    """Compute the mean/median and lower and upper bounds of an experiment result."""
    # Swap the trial and step axes (trial, step, user --> step, trial, user)
    arr = np.swapaxes(arr, 0, 1)
    # Flatten the trial and user axes together.
    arr = arr.reshape(arr.shape[0], -1)
    if use_median:
        # Compute the medians and quartiles of the means for each step.
        centers = np.median(arr, axis=1)
        upper_bounds = np.quantile(arr, 0.75, axis=1)
        lower_bounds = np.quantile(arr, 0.25, axis=1)
    else:
        # Compute the means and standard deviations of the means for each step.
        centers = arr.mean(axis=1)
        # Use Bessel's correction here.
        stds = arr.std(axis=1, ddof=1) / np.sqrt(arr.shape[1] - 1)
        # Compute the 95% confidence intervals using the CLT.
        upper_bounds = centers + 2 * stds
        lower_bounds = centers - 2 * stds
    if bound_zero:
        lower_bounds = np.maximum(lower_bounds, 0)

    return centers, lower_bounds, upper_bounds


def compute_stats_s3(bucket,
                     data_dir_name,
                     env_name,
                     rec_names,
                     seeds,
                     use_ratings=True,
                     arr_func=None,
                     bound_zero=False,
                     load_dense=False):
    """Compute the mean/median and lower and upper bounds of an experiment result stored in S3."""
    if arr_func is None:
        assert len(rec_names) == 1
        def arr_func(ratings, predictions):
            if use_ratings:
                return [ratings[0]]
            else:
                return [predictions[0]]

    def get_mean_square_func(preprocess_func):
        def compute_means_square(**kwargs):
            arrs = preprocess_func(**kwargs)
            means = []
            squares = []
            for arr in arrs:
                means.append(arr.mean(axis=1))
                squares.append((arr ** 2).mean(axis=1))
            return means, squares, arrs[0].shape[1]
        return compute_means_square

    results = compute_across_trials_s3(bucket,
                                       data_dir_name,
                                       env_name,
                                       rec_names,
                                       seeds,
                                       get_mean_square_func(arr_func),
                                       load_dense=load_dense)
    means, squares, lengths = zip(*results)
    stds = np.std(means, axis=0, ddof=1)
    means = np.average(means, axis=0, weights=lengths)
    squares = np.average(squares, axis=0, weights=lengths)
    variances = squares - means ** 2

    # Apply Bessel's correction.
    num_samples = np.sum(lengths)
    variances = variances * num_samples / (num_samples - 1)

    # Compute the standard error of each sample mean.
    # stds = np.sqrt(variances / (100 - 1))
    stds /= np.sqrt(len(seeds) - 1)

    # Compute the 95% confidence intervals using the CLT.
    upper_bounds = means + 2 * stds
    lower_bounds = means - 2 * stds
    if bound_zero:
        lower_bounds = np.maximum(lower_bounds, 0)

    return means, lower_bounds, upper_bounds


def get_env_dataset(environment):
    """Get the initial ratings of an environment.
    The intent of this function is to create an original dataset from which a recommender's
    hyperparameters can be tuned. The returned dataset will be identical to the original data
    available to each recommender when calling run_env_experiment.
    """
    environment.seed((INIT_SEED, 0))
    return environment.reset()


def run_env_experiment(environments,
                       recommenders,
                       trial_seeds,
                       len_trial,
                       environment_names=None,
                       recommender_names=None,
                       bucket_name='recsys-eval',
                       data_dir=None,
                       overwrite=False):
    """Run repeated trials for a given list of recommenders on a list of environments.
    Parameters
    ----------
    environments : Environment
        The environments to run the experiments with.
    recommenders : list of Recommender
        The recommenders to run the experiments with.
    len_trial : int
        The number of steps to run each trial for.
    trial_seeds : list of int
        The seeds to run each trial with.
    environment_names : list of str
        The name under which each environment will be saved. If this is None
        each environment will be named according to the environment's property.
    recommender_names : list of str
        The name under which each recommender will be saved. If this is None
        each recommender will be named according to the environment's property.
    bucket_name : str
        The name of the S3 bucket to store the experiment results in. If this is None
        the results will not be saved.
    data_dir : str
        The name of the S3 directory under which to store the experiments. Can be None
        if bucket_name is also None.
    overwrite : bool
        Whether to re-run the experiment even if a matching S3 file is found.
    Returns
    -------
    ratings : np.ndarray
        The array of all ratings made by users throughout all trials. ratings[i, j, k, l, m]
        corresponds to the rating made by the m-th online user during the l-th step of the
        k-th trial for the j-th recommender on the i-th environment.
    predictions : np.ndarray
        The array of all predictions made by recommenders throughout all trials.
        predictions[i, j, k, l, m] corresponds to the prediction that the j-th recommender
        made on the rating of the m-th online user during the l-th step of the
        k-th trial for the aforementioned recommender while running the i-th environment.
        Note that if the recommender does not make predictions to make recommendations
        then that element will be np.nan.
    dense_ratings : np.ndarray
        The array of all ratings throughout all trials. ratings[i, j, k, l]
        corresponds to the dense ratings array across all user-item pairs during
        the l-th step of the k-th trial for the j-th recommender on the i-th environment.
    predictions : np.ndarray
        The array of dense predictions made by recommenders throughout all trials.
        corresponds to the dense predictions array across all user-item pairs during
        the l-th step of the k-th trial for the j-th recommender on the i-th environment.
        predictions[i, j, k, l] corresponds to the prediction that the j-th recommender
    """
    bucket = None
    if bucket_name is not None:
        bucket = boto3.resource('s3').Bucket(bucket_name)  # pylint: disable=no-member

    if environment_names is None:
        environment_names = rename_duplicates([env.name for env in environments])

    if recommender_names is None:
        recommender_names = rename_duplicates([rec.name for rec in recommenders])

    all_ratings = []
    all_predictions = []
    all_dense_ratings = []
    all_dense_predictions = []
    for env_name, environment in zip(environment_names, environments):
        print('Started experiments on environment:', env_name)
        initial_density, final_density, good_item_density = compute_experiment_density(len_trial,
                                                                                       environment)
        print('\tInitial density: {}%, Final density: {}%, '.format(100 * initial_density,
                                                                    100 * final_density) +
              'Good item density: {}%'.format(100 * good_item_density))

        all_ratings.append([])
        all_predictions.append([])
        all_dense_ratings.append([])
        all_dense_predictions.append([])
        for rec_name, recommender in zip(recommender_names, recommenders):
            print('Running trials for recommender:', rec_name)
            all_ratings[-1].append([])
            all_predictions[-1].append([])
            all_dense_ratings[-1].append([])
            all_dense_predictions[-1].append([])
            for seed in trial_seeds:
                print('Running trial with seed:', seed)
                dir_name = s3_experiment_dir_name(data_dir, env_name, rec_name, seed)
                ratings, predictions, dense_ratings, dense_predictions = run_trial(
                    environment, recommender, len_trial, seed, bucket, dir_name, overwrite)
                all_ratings[-1][-1].append(ratings)
                all_predictions[-1][-1].append(predictions)
                all_dense_ratings[-1][-1].append(dense_ratings)
                all_dense_predictions[-1][-1].append(dense_predictions)

    # Convert all lists to arrays.
    all_ratings = np.array(all_ratings)
    all_predictions = np.array(all_predictions)
    all_dense_ratings = np.array(all_dense_ratings)
    all_dense_predictions = np.array(all_dense_predictions)

    return all_ratings, all_predictions, all_dense_ratings, all_dense_predictions


def run_trial(env,
              rec,
              len_trial,
              trial_seed,
              bucket=None,
              dir_name=None,
              overwrite=False):
    """Logic for running each trial.
    Parameters
    ----------
    env : Environment
        The environment to use for this trial.
    rec : Recommender
        The recommender to use for this trial.
    len_trial : int
        The number of recommendation steps to run the trial for.
    trial_seed : int
        Used to seed the dynamics of the environment.
    bucket : s3.Bucket
        The S3 bucket to store the experiment results into. If this is None the results
        will not be saved in S3.
    dir_name : str
        The S3 directory to save the trial results into. Can be None if bucket is also None.
    overwrite : bool
        Whether to re-run the experiment and overwrite the trial's saved data in S3.
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
    dense_ratings : np.ndarray
        The array of all dense ratings across each step. dense_ratings[i] is the
        array of all ratings that would have been made on round i for each user-item pair
        with all noise removed.
    dense_predictions : np.ndarray
        The array of all dense predictions across each step. dense_predictions[i] is the
        array of all predictions on round i for each user-item pair.
    """
    if not overwrite and s3_dir_exists(bucket, dir_name):
        print('Loading past results from S3 at directory:', dir_name)
        results = s3_load_trial(bucket, dir_name)
        return results[1:-1]

    # First generate the items and users to bootstrap the dataset.
    env.seed((INIT_SEED, trial_seed))
    items, users, ratings = env.reset()
    rec.reset(items, users, ratings)

    all_ratings = []
    all_predictions = []
    all_dense_ratings = []
    all_dense_predictions = []
    all_recs = []
    all_online_users = []
    all_env_snapshots = [copy.deepcopy(env)]
    # We have a separate variable for ratings.
    all_env_snapshots[-1]._ratings = None

    # Now recommend items to users.
    for _ in tqdm.autonotebook.tqdm(range(len_trial)):
        online_users = env.online_users()
        dense_predictions = rec.dense_predictions.flatten()
        recommendations, predictions = rec.recommend(online_users, num_recommendations=1)

        recommendations = recommendations.flatten()
        dense_ratings = np.clip(env.dense_ratings.flatten(), 1, 5)
        items, users, ratings, _ = env.step(recommendations)
        rec.update(users, items, ratings)

        # Account for the case where the recommender doesn't predict ratings.
        if predictions is None:
            predictions = np.ones_like(ratings) * np.nan
            dense_predictions = np.ones_like(dense_ratings) * np.nan

        # Save all relevant info.
        all_ratings.append([rating for rating, _ in ratings.values()])
        all_predictions.append(predictions.flatten())
        all_dense_ratings.append(dense_ratings)
        all_dense_predictions.append(dense_predictions)
        all_recs.append(recommendations)
        all_online_users.append(online_users)
        all_env_snapshots.append(copy.deepcopy(env))
        all_env_snapshots[-1]._ratings = None

    # Convert lists to numpy arrays
    all_ratings = np.array(all_ratings)
    all_predictions = np.array(all_predictions)
    all_dense_ratings = np.array(all_dense_ratings)
    all_dense_predictions = np.array(all_dense_predictions)
    all_recs = np.array(all_recs)
    all_online_users = np.array(all_online_users)

    # Save content to S3 if needed.
    if bucket is not None:
        print('Saving results to S3.')
        s3_save_trial(bucket,
                      dir_name,
                      env.name,
                      rec.name,
                      rec.hyperparameters,
                      all_ratings,
                      all_predictions,
                      all_dense_ratings,
                      all_dense_predictions,
                      all_recs,
                      all_online_users,
                      all_env_snapshots)

    # TODO: We might want to return the env snapshots too.
    return all_ratings, all_predictions, all_dense_ratings, all_dense_predictions


def compute_experiment_density(len_trial, environment, threshold=4):
    """Compute the rating density for the proposed experiment.
    Parameters
    ----------
    len_trial : int
        Length of trial.
    environment : Environment
        The environment to consider.
    threshold : int
        The threshold for a rating to be considered "good".
    Returns
    -------
    initial_density : float
        The initial rating matrix density.
    final_density : float
        The final rating matrix density.
    good_item_density : float
        The underlying density of good items in the environment.
    """
    # Initialize environment
    get_env_dataset(environment)
    total_num_ratings = len(environment.users) * len(environment.items)

    initial_density = environment._num_init_ratings / total_num_ratings
    num_ratings_per_it = len(environment._users) * environment._rating_frequency
    final_num_ratings = environment._num_init_ratings + len_trial * num_ratings_per_it
    final_density = final_num_ratings / total_num_ratings

    num_good_ratings = np.sum(environment.dense_ratings > threshold)
    good_item_density = num_good_ratings / total_num_ratings

    return initial_density, final_density, good_item_density


class ModelTuner:
    """The tuner allows for easy tuning.
    Provides functionality for n-fold cross validation to
    assess the performance of various model parameters.
    Parameters
    ----------
    data : triple of iterables
        The (user, items, ratings) data.
    default_params : dict
        Default model parameters.
    recommender_class : class Recommender
        The class of the recommender on which we wish to tune parameters.
    n_fold : int, optional
        The number of folds for cross validation.
    verbose : bool, optional
        Mode for printing results, defaults to True.
    bucket_name : str
        The name of the S3 bucket to store the tuning logs in. If this is None
        the results will not be saved.
    data_dir : str
        The name of the S3 directory under which to store the tuning logs. Can be None
        if bucket_name is also None.
    environment_name : str
        The name of the environment snapshot on which we are tuning the recommender. Can be None
        if bucket_name is also None.
    recommender_name : str
        The name of the recommender for which we are storing the tuning logs. Can be None
        if bucket_name is also None.
    overwrite : bool
        Whether to overwrite tuning logs in S3 if they already exist.
    """

    def __init__(self,
                 data,
                 default_params,
                 recommender_class,
                 n_fold=5,
                 verbose=True,
                 bucket_name=None,
                 data_dir=None,
                 environment_name=None,
                 recommender_name=None,
                 overwrite=False,
                 use_mse=True):
        """Create a model tuner."""
        self.users, self.items, self.ratings = data
        self.default_params = default_params
        self.num_users = len(self.users)
        self.num_items = len(self.items)
        self.verbose = verbose
        self.recommender_class = recommender_class
        self.bucket = None
        self.data_dir = data_dir
        self.environment_name = environment_name
        self.recommender_name = recommender_name
        self.overwrite = overwrite
        self.num_evaluations = 0
        self.use_mse = use_mse

        if bucket_name is not None:
            if self.data_dir is None:
                raise ValueError('data_dir can not be None when bucket_name is not None.')
            if self.environment_name is None:
                raise ValueError('environment_name can not be None when bucket_name is not None.')
            if self.recommender_name is None:
                raise ValueError('recommender_name can not be None when bucket_name is not None.')
            self.bucket = boto3.resource('s3').Bucket(bucket_name)  # pylint: disable=no-member

        self._generate_n_folds(n_fold)

    def _generate_n_folds(self, n_fold):
        """Generate indices for n folds."""
        indices = np.random.permutation(len(self.ratings))
        size_fold = len(self.ratings) // n_fold
        self.train_test_folds = []
        for i in range(n_fold):
            test_ind = indices[i*size_fold:(i+1)*size_fold]
            train_ind = np.append(indices[:i*size_fold], indices[(i+1)*size_fold:])
            self.train_test_folds.append((train_ind, test_ind))

    def evaluate(self, params):
        """Train and evaluate a model for parameter setting."""
        # constructing model with given parameters
        defaults = {key: self.default_params[key] for key in self.default_params.keys()
                    if key not in params.keys()}
        recommender = self.recommender_class(**defaults, **params)
        metrics = []
        if self.verbose:
            print('Evaluating:', params)
        for i, fold in enumerate(self.train_test_folds):
            if self.verbose:
                print('Fold {}/{}, '.format(i+1, len(self.train_test_folds)),
                      end='')
            train_ind, test_ind = fold

            # splitting data dictionaries
            keys = list(self.ratings.keys())
            ratings_test = {key: self.ratings[key] for key in [keys[i] for i in test_ind]}
            ratings_train = {key: self.ratings[key] for key in [keys[i] for i in train_ind]}

            recommender.reset(self.users, self.items, ratings_train)

            # constructing test inputs
            ratings_to_predict = []
            true_ratings = []
            for user, item in ratings_test.keys():
                true_r, context = self.ratings[(user, item)]
                ratings_to_predict.append((user, item, context))
                true_ratings.append(true_r)
            predicted_ratings = recommender.predict(ratings_to_predict)

            if self.use_mse:
                mse = np.mean((predicted_ratings - true_ratings)**2)
                if self.verbose:
                    print('mse={}, rmse={}'.format(mse, np.sqrt(mse)))
                metrics.append(mse)
            else:
                # Note that this is not quite a traditional NDCG
                # normally we would consider users individually
                # this computation lumps all predictions together.
                def get_ranks(array):
                    array = np.array(array)
                    temp = array.argsort()
                    ranks = np.empty_like(temp)
                    ranks[temp] = np.arange(len(array))
                    return len(ranks) - ranks
                def get_dcg(ranks, relevances, cutoff=5):
                    dcg = 0
                    for rank, relevance in zip(ranks, relevances):
                        if rank <= cutoff:
                            dcg += relevance / np.log2(rank+1)
                    return dcg
                user_preds = collections.defaultdict(list)
                user_true = collections.defaultdict(list)
                for (user, _, _), pred, true in zip(ratings_to_predict, predicted_ratings, true_ratings):
                    user_preds[user].append(pred)
                    user_true[user].append(true)
                cutoff = 20 # int(len(true_ratings) / 5)
                idcgs = []
                dcgs = []
                ndcgs = []
                for user in user_preds:
                    pred = user_preds[user]
                    true = user_true[user]
                    idcg = get_dcg(get_ranks(true), true, cutoff=cutoff)
                    dcg = get_dcg(get_ranks(pred), true, cutoff=cutoff)
                    dcgs.append(dcg)
                    ndcgs.append(dcg / idcg)
                dcg = np.mean(dcgs)
                ndcg = np.mean(ndcgs)
                if self.verbose:
                    print('dcg={}, ndcg={}'.format(dcg, ndcg))
                metrics.append(ndcg)
            

        if self.verbose:
            print('Average Metric:', np.mean(metrics))
        return np.array(metrics)

    def evaluate_grid(self, **params):
        """Train over a grid of parameters."""
        def recurse_grid(fixed_params, grid_params):
            if len(grid_params) == 0:
                result = fixed_params
                result['metric'] = self.evaluate(fixed_params)
                result['average_metric'] = np.mean(result['metric'])
                return [result]

            curr_param, curr_values = list(grid_params.items())[0]
            new_grid_params = dict(grid_params)
            del new_grid_params[curr_param]
            results = []
            for value in curr_values:
                results += recurse_grid({**fixed_params, curr_param: value}, new_grid_params)
            return results

        results = recurse_grid({}, params)
        results = pd.DataFrame(results)
        if self.bucket is not None:
            self.s3_save(results, params)
        self.num_evaluations += 1
        return results

    def s3_save(self, results, params):
        """Save the current hyperparameter tuning results to S3."""
        dir_name = os.path.join(self.data_dir, self.environment_name, self.recommender_name,
                                'tuning', 'evaluation_' + str(self.num_evaluations), '')
        if s3_dir_exists(self.bucket, dir_name) and not self.overwrite:
            if self.verbose:
                print('Directory:', dir_name, 'already exists. Results will not be saved to S3.')
            return

        if self.verbose:
            print('Saving to S3 directory:', dir_name)
        info = {
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'git branch': git_branch(),
            'git hash': git_hash(),
            'git username': git_username(),
            'recommender': self.recommender_name,
        }
        serialize_and_put(self.bucket, dir_name, 'info', info, use_json=True)
        serialize_and_put(self.bucket, dir_name, 'params', params, use_json=True)
        put_dataframe(self.bucket, dir_name, 'results', results)


def rename_duplicates(old_list):
    """Append a number to each element in a list of strings based on the number of duplicates."""
    count = collections.defaultdict(int)
    new_list = []
    for x in old_list:
        new_list.append(x + '_' + str(count[x]))
        count[x] += 1
    return new_list


def git_username():
    """Get the git username of the person running the code."""
    return subprocess.check_output(['git', 'config', 'user.name']).decode('ascii').strip()


def git_hash():
    """Get the current git hash of the code."""
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def git_branch():
    """Get the current git branch of the code."""
    return subprocess.check_output(['git', 'rev-parse',
                                    '--abbrev-ref', 'HEAD']).decode('ascii').strip()


def compute_across_trials_s3(bucket,
                             data_dir,
                             env_name,
                             rec_names,
                             seeds,
                             func,
                             load_dense=False):
    """Apply func to all the trials of an experiment and return a list of func's return values.
    This function loads one trial at a time to prevent memory issues.
    """
    results = []
    for seed in seeds:
        all_ratings = []
        all_predictions = []
        for rec_name in rec_names:
            dir_name = s3_experiment_dir_name(data_dir, env_name, rec_name, seed)
            (_, ratings, predictions,
             dense_ratings, dense_predictions, _) = s3_load_trial(bucket,
                                                                  dir_name,
                                                                  load_dense=load_dense)
            if load_dense:
                ratings = dense_ratings
                predictions = dense_predictions
            all_ratings.append(ratings)
            all_predictions.append(predictions)
        results.append(func(ratings=all_ratings,
                            predictions=all_predictions))

        # Make these variables out of scope so they can be garbage collected.
        ratings = None
        predictions = None
        dense_ratings = None
        dense_predictions = None
        all_ratings = None
        all_predictions = None

    return results


def s3_experiment_dir_name(data_dir, env_name, rec_name, trial_seed):
    """Get the directory name that corresponds to a given trial."""
    if data_dir is None:
        return None
    return os.path.join(data_dir, env_name, rec_name, 'trials', 'seed_' + str(trial_seed), '')


def s3_dir_exists(bucket, dir_name):
    """Check if a directory exists in S3."""
    if bucket is None:
        return False

    # We can't use len here so do this instead.
    exists = False
    for _ in bucket.objects.filter(Prefix=dir_name):
        exists = True
        break

    return exists


def s3_save_trial(bucket,
                  dir_name,
                  env_name,
                  rec_name,
                  rec_hyperparameters,
                  ratings,
                  predictions,
                  dense_ratings,
                  dense_predictions,
                  recommendations,
                  online_users,
                  env_snapshots):
    """Save a trial in s3 within the given directory."""
    info = {
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'environment': env_name,
        'git branch': git_branch(),
        'git hash': git_hash(),
        'git username': git_username(),
        'recommender': rec_name,
    }
    serialize_and_put(bucket, dir_name, 'info', info, use_json=True)
    serialize_and_put(bucket, dir_name, 'rec_hyperparameters', rec_hyperparameters, use_json=True)
    serialize_and_put(bucket, dir_name, 'ratings', ratings)
    serialize_and_put(bucket, dir_name, 'predictions', predictions)
    serialize_and_put(bucket, dir_name, 'dense_ratings', dense_ratings)
    serialize_and_put(bucket, dir_name, 'dense_predictions', dense_predictions)
    serialize_and_put(bucket, dir_name, 'recommendations', recommendations)
    serialize_and_put(bucket, dir_name, 'online_users', online_users)
    serialize_and_put(bucket, dir_name, 'env_snapshots', env_snapshots)


def s3_load_trial(bucket, dir_name, load_dense=True):
    """Load a trial saved in a given directory within S3."""
    def get_and_unserialize(name, use_json=False):
        file_name = os.path.join(dir_name, name)
        if use_json:
            file_name = file_name + '.json'
        else:
            file_name = file_name + '.pickle'

        with open(TEMP_FILE_NAME, 'wb') as temp_file:
            bucket.download_fileobj(Key=file_name, Fileobj=temp_file)

        with open(TEMP_FILE_NAME, 'rb') as temp_file:
            if use_json:
                obj = json.load(temp_file)
            else:
                obj = pickle.load(temp_file)
        os.remove(TEMP_FILE_NAME)

        return obj

    ratings = get_and_unserialize('ratings')
    predictions = get_and_unserialize('predictions')
    rec_hyperparameters = get_and_unserialize('rec_hyperparameters', use_json=True)
    if load_dense:
        dense_ratings = get_and_unserialize('dense_ratings')
        dense_predictions = get_and_unserialize('dense_predictions')
        env_snapshots = get_and_unserialize('env_snapshots')
    else:
        dense_ratings = None
        dense_predictions = None
        env_snapshots = None

    return (rec_hyperparameters, ratings, predictions,
            dense_ratings, dense_predictions, env_snapshots)


def serialize_and_put(bucket, dir_name, name, obj, use_json=False):
    """Serialize an object and upload it to S3."""
    file_name = os.path.join(dir_name, name)
    with open(TEMP_FILE_NAME, 'wb') as temp_file:
        if use_json:
            json.dump(obj, codecs.getwriter('utf-8')(temp_file),
                      sort_keys=True, indent=4)
            file_name = file_name + '.json'
        else:
            pickle.dump(obj, temp_file, protocol=4)
            file_name = file_name + '.pickle'

    with open(TEMP_FILE_NAME, 'rb') as temp_file:
        bucket.upload_fileobj(Key=file_name, Fileobj=temp_file,
                              ExtraArgs={'GrantFullControl': ID_STR})

    os.remove(TEMP_FILE_NAME)


def put_dataframe(bucket, dir_name, name, dataframe):
    """Upload a dataframe to S3 as a csv file."""
    with io.StringIO() as stream:
        dataframe.to_csv(stream)
        csv_str = stream.getvalue()

    with io.BytesIO() as stream:
        stream.write(csv_str.encode('utf-8'))
        stream.seek(0)
        file_name = os.path.join(dir_name, name + '.csv')
        bucket.upload_fileobj(Key=file_name, Fileobj=stream,
                              ExtraArgs={'GrantFullControl': ID_STR})
