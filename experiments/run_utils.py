"""A utility module for running experiments."""
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
import tqdm.autonotebook


# The random seed that defines the initial state of each environment.
INIT_SEED = 0


def plot_ratings_mses(ratings,
                      predictions,
                      labels,
                      num_init_ratings=None):
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
    num_init_ratings : int
        The number of ratings initially available to recommenders. If set to None
        the function will plot with an x-axis based on round number.

    """
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

    if num_init_ratings is not None:
        x_vals = num_init_ratings + ratings.shape[3] * np.arange(ratings.shape[2])
    else:
        x_vals = np.arange(ratings.shape[2])

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


def get_env_dataset(environment):
    """Get the initial ratings of an environment.

    The intent of this function is to create an original dataset from which a recommender's
    hyperparameters can be tuned. The returned dataset will be identical to the original data
    available to each recommender when calling run_env_experiment.

    """
    env.seed((INIT_SEED, 0))
    return env.reset()


def run_env_experiment(environments,
                       recommenders,
                       n_trials,
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
    n_trials : int
        The number of trials to run for each recommender.
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
            for i in range(n_trials):
                print('Running trial:', i)
                dir_name = s3_dir_name(data_dir, env_name, rec_name, i)
                ratings, predictions, dense_ratings, dense_predictions = run_trial(
                    environment, recommender, len_trial, i, bucket, dir_name, overwrite)
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
              trial_number,
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
    trial_number : int
        The index of the trial. Used to seed the dynamics of the environment.
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
        print('Loading past results from S3.')
        results = s3_load_trial(bucket, dir_name)
        return results[1:-1]

    # First generate the items and users to bootstrap the dataset.
    env.seed((INIT_SEED, trial_number))
    items, users, ratings = env.reset()
    rec.reset(items, users, ratings)

    all_ratings = []
    all_predictions = []
    all_dense_ratings = []
    all_dense_predictions = []
    all_env_snapshots = [pickle.dumps(env)]
    user_item = []
    for i in range(len(env.users)):
        for j in range(len(env.items)):
            user_item.append((i, j, np.zeros(0)))

    # Now recommend items to users.
    for _ in tqdm.autonotebook.tqdm(range(len_trial)):
        online_users = env.online_users()
        recommendations, predictions = rec.recommend(online_users, num_recommendations=1)
        recommendations = recommendations.flatten()
        dense_ratings = np.clip(env.dense_ratings.flatten(), 1, 5)
        items, users, ratings, _ = env.step(recommendations)

        # Account for the case where the recommender doesn't predict ratings.
        if predictions is None:
            predictions = np.ones_like(ratings) * np.nan
            dense_predictions = np.ones_like(dense_ratings) * np.nan
        else:
            predictions = predictions.flatten()
            dense_predictions = rec.predict(user_item)

        # Save all relevant info.
        all_ratings.append([rating for rating, _ in ratings.values()])
        all_predictions.append(predictions)
        all_dense_ratings.append(dense_ratings)
        all_dense_predictions.append(dense_predictions)
        all_env_snapshots.append(copy.deepcopy(env))

        rec.update(users, items, ratings)

    # Convert lists to numpy arrays
    all_ratings = np.array(all_ratings)
    all_predictions = np.array(all_predictions)
    all_dense_ratings = np.array(all_dense_ratings)
    all_dense_predictions = np.array(all_dense_predictions)

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
                      all_env_snapshots)

    # TODO: We might want to return the env snapshots too.
    return all_ratings, all_predictions, all_dense_ratings, all_dense_predictions


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
    n_fold : int, optional
        The number of folds for cross validation.
    verbose : bool, optional
        Mode for printing results, defaults to True.

    """

    def __init__(self, data, default_params, recommender_object, n_fold=5, verbose=True):
        """Create a model tuner."""
        self.users, self.items, self.ratings = data
        self.default_params = default_params
        self.num_users = len(self.users)
        self.num_items = len(self.items)
        self.verbose = verbose
        self._generate_n_folds(n_fold)
        self.recommender_object = recommender_object

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
        recommender = self.recommender_object(**defaults, **params)
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
        return mses

    def evaluate_list(self, param_tag_list):
        """Train over list of parameters."""
        res_dict = {}
        for tag, params in param_tag_list:
            mses = self.evaluate(params)
            res_dict[tag] = mses
        return res_dict


def rename_duplicates(old_list):
    """Append a number to each element in a list of strings based on the number of duplicates."""
    count = collections.defaultdict(int)
    new_list = []
    for x in old_list:
        new_list.append(x + '_' + str(count[x]))
        count[x] += 1
    return new_list


def git_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def git_branch():
    return subprocess.check_output(['git', 'rev-parse',
                                    '--abbrev-ref', 'HEAD']).decode('ascii').strip()


def s3_dir_name(data_dir, env_name, rec_name, trial_number):
    """Get the directory name that corresponds to a given trial."""
    if data_dir is None:
        return None
    return os.path.join(data_dir, env_name, rec_name, 'trial_' + str(trial_number), '')


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
                  env_snapshots):
    """Save a trial in s3 within the given directory."""
    def serialize_and_put(name, obj, use_json=False):
        file_name = os.path.join(dir_name, name)
        if use_json:
            serialized_obj = json.dumps(obj, sort_keys=True, indent=4)
            file_name  = file_name + '.json'
        else:
            serialized_obj = pickle.dumps(obj)
            file_name  = file_name + '.pickle'
        bucket.put_object(Key=file_name, Body=serialized_obj)

    info = {
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'environment': env_name,
        'recommender': rec_name,
        'git hash': git_hash(),
        'git branch': git_branch(),
    }
    serialize_and_put('info', info, use_json=True)
    serialize_and_put('rec_hyperparameters', rec_hyperparameters, use_json=True)
    serialize_and_put('ratings', ratings)
    serialize_and_put('predictions', predictions)
    serialize_and_put('dense_ratings', dense_ratings)
    serialize_and_put('dense_predictions', dense_predictions)
    serialize_and_put('env_snapshots', env_snapshots)


def s3_load_trial(bucket, dir_name):
    """Load a trial saved in a given directory within s3."""
    def get_and_unserialize(name, use_json=False):
        file_name = os.path.join(dir_name, name)
        if use_json:
            file_name  = file_name + '.json'
        else:
            file_name  = file_name + '.pickle'
        with io.BytesIO() as stream:
            bucket.download_fileobj(Key=file_name, Fileobj=stream)
            serialized_obj = stream.getvalue()
        if use_json:
            obj = json.loads(serialized_obj)
        else:
            obj = pickle.loads(serialized_obj)
        return obj

    rec_hyperparameters = get_and_unserialize('rec_hyperparameters', use_json=True)
    ratings = get_and_unserialize('ratings')
    predictions = get_and_unserialize('predictions')
    dense_ratings = get_and_unserialize('dense_ratings')
    dense_predictions = get_and_unserialize('dense_predictions')
    env_snapshots = get_and_unserialize('env_snapshots')

    return (rec_hyperparameters, ratings, predictions,
            dense_ratings, dense_predictions, env_snapshots)
