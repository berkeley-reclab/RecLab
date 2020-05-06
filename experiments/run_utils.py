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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm.autonotebook


# The random seed that defines the initial state of each environment.
INIT_SEED = 0
# The name of the file temporarily created for uploads to S3.
TEMP_FILE_NAME = 'temp.out'


def plot_ratings_mses(ratings,
                      predictions,
                      labels,
                      summary_type='mean',
                      num_init_ratings=None, threshold=10):
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
    summary_type : str
        Which time of summary statistics: either 'mean' for mean and std deviation
        or 'median' for median and quartiles.
    num_init_ratings : int
        The number of ratings initially available to recommenders. If set to None
        the function will plot with an x-axis based on round number.
    threshold: float
        The threshold filtering on the predictions, predictions larger than it will be set to 0.
        default is 10

    """
    if summary_type == 'mean':
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
    elif summary_type == 'median':
        def get_stats(arr):
            # Swap the trial and step axes.
            arr = np.swapaxes(arr, 0, 1)
            # Flatten the trial and user axes together.
            arr = arr.reshape(arr.shape[0], -1)
            # Compute the medians and quartiles of the means for each step.
            meds = np.median(arr, axis=1)
            # Compute the 95% confidence intervals using the CLT.
            upper_bounds = np.quantile(arr, 0.75, axis=1)
            lower_bounds = np.quantile(arr, 0.25, axis=1)
            return meds, lower_bounds, upper_bounds
    else:
        assert False, "Invalid summary_type"

    if num_init_ratings is not None:
        x_vals = num_init_ratings + ratings.shape[3] * np.arange(ratings.shape[2])
    else:
        x_vals = np.arange(ratings.shape[2])

    # Setting the predictions for a user/item that has no ratings in the training data to 0.
    predictions[predictions > threshold] = 0

    plt.figure(figsize=[9, 4])
    plt.subplot(1, 2, 1)
    for recommender_ratings, label in zip(ratings, labels):
        means, lower_bounds, upper_bounds = get_stats(recommender_ratings)
        plt.plot(x_vals, means, label=label)
        plt.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.1)
    plt.xlabel('# ratings')
    plt.ylabel('Mean Rating')
    plt.legend()

    plt.subplot(1, 2, 2)
    squared_diffs = (ratings - predictions) ** 2
    for recommender_squared_diffs, label in zip(squared_diffs, labels):
        mse, lower_bounds, upper_bounds = get_stats(recommender_squared_diffs)
        # Transform the MSE into the RMSE and correct the associated intervals.
        rmse = np.sqrt(mse)
        lower_bounds = np.sqrt(lower_bounds)
        upper_bounds = np.sqrt(upper_bounds)
        plt.plot(x_vals, rmse, label=label)
        plt.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.1)
    plt.xlabel('# ratings')
    plt.ylabel('RMSE')
    plt.legend()
    plt.tight_layout()
    plt.show()


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
            idx = labels.index('perfert')
            perfect_ratings = ratings[idx]
        else:
            print('no ratings from the perfect recommeder')
            return

    def get_regret_stats(arr):
        # Swap the trial and step axes (trial, step, user --> step, trial, user)
        arr = np.swapaxes(arr, 0, 1)
        # Flatten the trial and user axes together.
        arr = arr.reshape(arr.shape[0], -1)
        # Compute the commulative regret.
        arr = arr.cumsum(axis=1)
        # Compute the means and standard deviations of the means for each step.
        means = arr.mean(axis=1)
        # Use Bessel's correction here.
        stds = arr.std(axis=1) / np.sqrt(arr.shape[1] - 1)
        # Compute the 95% confidence intervals using the CLT.
        upper_bounds = means + 2 * stds
        lower_bounds = means - 2 * stds
        return means, lower_bounds, upper_bounds

    if num_init_ratings is not None:
        x_vals = num_init_ratings + ratings.shape[3] * np.arange(ratings.shape[2])
    else:
        x_vals = np.arange(ratings.shape[2])

    plt.figure(figsize=[5, 4])
    for recommender_ratings, label in zip(ratings, labels):
        # Plot the regret for the recommenders that are not perfect.
        if label != 'perfect':
            regrets = perfect_ratings - recommender_ratings
            mean_regrets, lower_bounds, upper_bounds = get_regret_stats(regrets)
            # Plotting the regret over steps and correct the associated intervals.
            plt.plot(x_vals, mean_regrets, label=label)
            plt.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.1)
    plt.xlabel('# ratings')
    plt.ylabel('Regret')
    plt.legend()
    plt.tight_layout()
    plt.show()


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
    # We have a seperate variable for ratings.
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
    """ Compute the rating density for the proposed experiment.

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
                 overwrite=False):
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
        mses = []
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

            mse = np.mean((predicted_ratings - true_ratings)**2)
            if self.verbose:
                print('mse={}'.format(mse))
            mses.append(mse)

        if self.verbose:
            print('Average MSE:', np.mean(mses))
        return np.array(mses)

    def evaluate_grid(self, **params):
        """Train over a grid of parameters."""
        def recurse_grid(fixed_params, grid_params):
            if len(grid_params) == 0:
                result = fixed_params
                result['mse'] = self.evaluate(fixed_params)
                result['average_mse'] = np.mean(result['mse'])
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


def s3_load_trial(bucket, dir_name):
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

    rec_hyperparameters = get_and_unserialize('rec_hyperparameters', use_json=True)
    ratings = get_and_unserialize('ratings')
    predictions = get_and_unserialize('predictions')
    dense_ratings = get_and_unserialize('dense_ratings')
    dense_predictions = get_and_unserialize('dense_predictions')
    env_snapshots = get_and_unserialize('env_snapshots')

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
        bucket.upload_fileobj(Key=file_name, Fileobj=temp_file)

    os.remove(TEMP_FILE_NAME)


def put_dataframe(bucket, dir_name, name, dataframe):
    """Upload a dataframe to S3 as a csv file."""
    with io.StringIO() as stream:
        dataframe.to_csv(stream)
        csv_str = stream.getvalue()

    with io.BytesIO() as stream:
        stream.write(csv_str.encode('utf-8'))
        file_name = os.path.join(dir_name, name + '.csv')
        bucket.upload_fileobj(Key=file_name, Fileobj=stream)
