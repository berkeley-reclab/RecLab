"""A utility module for running experiments."""
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_ratings_mses(ratings,
                      predictions,
                      num_init_ratings,
                      labels):
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
    num_init_ratings : int
        The number of ratings initially available to recommenders.
    labels : list of str
        The name of each recommender.

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

    plt.figure(figsize=[9, 4])
    x_vals = num_init_ratings + ratings.shape[3] * np.arange(ratings.shape[2])
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


def run_env_experiment(environments,
                       recommenders,
                       n_trials,
                       len_trial,
                       exp_dirname,
                       data_filename,
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
    exp_dirname : str
        The directory in which to save or load the experiment results.
    data_filename : str
        The name of the file in which to save or load the experiment results.
    overwrite : bool
        Whether to re-run the experiment even if a matching file is found.

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

    """
    datadirname = os.path.join('data', exp_dirname)
    os.makedirs(datadirname, exist_ok=True)

    filename = os.path.join(datadirname, data_filename)
    if not os.path.exists(filename) or overwrite:
        all_ratings = []
        all_predictions = []
        for environment in environments:
            all_ratings.append([])
            all_predictions.append([])
            for recommender in recommenders:
                all_ratings[-1].append([])
                all_predictions[-1].append([])
                for _ in range(n_trials):
                    ratings, predictions = run_trial(environment, recommender, len_trial)
                    all_ratings[-1][-1].append(ratings)
                    all_predictions[-1][-1].append(predictions)
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
    for _ in range(len_trial):
        online_users = env.online_users()
        recommendations, predictions = recommender.recommend(online_users, num_recommendations=1)
        recommendations = recommendations.flatten()
        items, users, ratings, _ = env.step(recommendations)
        recommender.update(users, items, ratings)
        ratings = [rating for rating, _ in ratings.values()]
        all_ratings.append(ratings)
        if predictions is None:
            predictions = np.ones(ratings.shape) * np.nan
        else:
            predictions = predictions.flatten()
        all_predictions.append(predictions)

    return all_ratings, all_predictions


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
