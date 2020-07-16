"""A utility module for plotting experiments."""
import matplotlib.pyplot as plt
import numpy as np

import s3


def plot_novelty_s3(bucket_name,
                    data_dir,
                    env_name,
                    rec_names,
                    seeds,
                    num_init_ratings,
                    labels):
    """Plot novelty for multiple recommenders using data from S3.

    Parameters
    ----------
    bucket_name : str
        The bucket in which the experiment data is saved.
    data_dir: str
        The name of the directory in which the experiment data is saved.
    env_name : str
        The name of the environment for which we are plotting the results.
    rec_names : str
        The names of the recommenders for computing novelty.
    seeds : list of int
        The trial seeds across which we are averaging results.
    num_init_ratings : int
        The number of ratings initially available to recommenders. If set to None
        the function will plot with an x-axis based on the timestep.
    labels : list of str
        The name of each recommender.

    """
    def compute_stats(novelty):
        means = novelty.mean(axis=0)
        variances = novelty.var(axis=0)
        num_samples = len(seeds)
        variances = variances * num_samples / (num_samples - 1)
        stds = np.sqrt(variances / (num_samples - 1))
        upper_bounds = means + 2 * stds
        lower_bounds = np.maximum(means - 2 * stds, 0)
        return means, upper_bounds, lower_bounds

    results = []
    for rec_name in rec_names:
        novelty = []
        for seed in seeds:
            dir_name = s3.experiment_dir_name(data_dir, env_name, rec_name, seed)
            recommendations = s3.get_and_unserialize(
                bucket_name, dir_name, 'recommendations')
            online_users = s3.get_and_unserialize(
                bucket_name, dir_name, 'online_users')
            env = s3.get_and_unserialize(bucket_name, dir_name, 'env_snapshots')[0]
            # Since some experiments were run before the user sampling PR.
            env._user_dist_choice = 'uniform'
            novelty.append(compute_novelty(recommendations, online_users, env))
        novelty = list(compute_stats(np.array(novelty)))
        results.append(novelty)

    plt.figure(figsize=(18, 6))
    for i in range(len(rec_names)):
        x_vals = [num_init_ratings + (i * recommendations.shape[1])
                  for i in range(recommendations.shape[0])]
        plt.plot(x_vals, results[i][0], label=labels[i])
        plt.fill_between(x_vals, results[i][2], results[i][1], alpha=0.1)

    plt.xlabel('# of ratings')
    plt.ylabel('Novelty')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return results


def compute_novelty(recommendations, online_users, env):
    """Compute novelty based on a list of recommendations and users.

    Parameters
    ---------
    recommendations : np.ndarray
        The array of recommendations for a single trial run for a single recommender.
    online users : list of dictionaries
        The list of online users at each timestep.
    env : Environment object
        Saved environment from experiment for retrieving number of items and users.

    """
    num_users = env._num_users
    num_items = env._num_items
    seen = dict()
    novelty = []
    for i in range(num_items):
        seen[i] = set()
    for i in range(recommendations.shape[0]):
        novelty_t = 0
        for item, user in zip(recommendations[i], list(online_users[i].keys())):
            # if an item has never been before, set an arbitrary p_i
            if len(seen[item]) == 0:
                p_i = (1 / num_users)
            else:
                p_i = len(seen[item]) / num_users
            novelty_t += -1 * np.log2(p_i)
            # normalize novelty to between 0 and 1
            novelty_t /= (recommendations.shape[1])
        novelty.append(novelty_t)
        for item, user in zip(recommendations[i], list(online_users[i].keys())):
            seen[item].add(user)
    return novelty


def plot_ratings_mses(ratings,
                      predictions,
                      labels,
                      use_median=False,
                      num_init_ratings=None,
                      threshold=10,
                      title=None):
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
    title: tuple of str
        The titles to label each plot with.

    """
    if title is None:
        title = ('', '')

    if num_init_ratings is not None:
        x_vals = num_init_ratings + ratings.shape[3] * np.arange(ratings.shape[2])
    else:
        x_vals = np.arange(ratings.shape[2])

    # Setting the predictions for a user/item that has no ratings in the training data to 0.
    predictions[predictions > threshold] = 0

    plt.figure(figsize=[9, 4])
    plt.subplot(1, 2, 1)
    for recommender_ratings, label in zip(ratings, labels):
        means, lower_bounds, upper_bounds = compute_stats(recommender_ratings,
                                                          bound_zero=True,
                                                          use_median=use_median)
        plt.plot(x_vals, means, label=label)
        plt.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.1)
    plt.xlabel('# ratings')
    plt.ylabel('Mean Rating')
    plt.title(title[0])
    plt.legend()

    plt.subplot(1, 2, 2)
    squared_diffs = (ratings - predictions) ** 2
    for recommender_squared_diffs, label in zip(squared_diffs, labels):
        mse, lower_bounds, upper_bounds = compute_stats(recommender_squared_diffs,
                                                        bound_zero=True,
                                                        use_median=use_median)
        # Transform the MSE into the RMSE and correct the associated intervals.
        rmse = np.sqrt(mse)
        lower_bounds = np.sqrt(lower_bounds)
        upper_bounds = np.sqrt(upper_bounds)
        plt.plot(x_vals, rmse, label=label)
        plt.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.1)
    plt.xlabel('# ratings')
    plt.ylabel('RMSE')
    plt.title(title[1])
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_coverage_s3(rec_names,
                    bucket_name,
                    data_dir,
                    env_name,
                    seeds):
    """Compute coverage based on data stored in S3.

    Parameters
    ----------
    rec_names : str
        The names of the recommenders for computing novelty.
    bucket_name : str
        The bucket in which the experiment data is saved.
    data_dir: str
        The name of the directory in which the experiment data is saved.
    env_name : str
        The name of the environment for which we are plotting the results.
    seeds : list of int
        The trial seeds across which we are averaging results.

    """
    results = []
    for rec_name in rec_names:
        coverage = []
        for seed in seeds:
            dir_name = s3.experiment_dir_name(data_dir, env_name, rec_name, seed)
            recommendations = s3.get_and_unserialize(
                bucket_name, dir_name, 'recommendations')
            rec_coverage = np.mean([len(set(rec)) for rec in recommendations])
            coverage.append(rec_coverage)
        results.append(coverage)
    return results


def plot_coverage_s3(rec_names,
                     bucket_name,
                     data_dir,
                     env_name,
                     seeds,
                     num_init_ratings=None):
    """Plot coverage based on list of recommenders and data from S3.

    Parameters
    ----------
    rec_names : str
        The names of the recommenders for computing novelty.
    len_trial : int
        The number of timesteps per trial.
    bucket_name : str
        The bucket in which the experiment data is saved.
    data_dir: str
        The name of the directory in which the experiment data is saved.
    env_name : str
        The name of the environment for which we are plotting the results.
    seeds : list of int
        The trial seeds across which we are averaging results.
    num_init_ratings : int
        The number of ratings initially available to recommenders. If set to None
        the function will plot with an x-axis based on the timestep.

    """
    results = []
    for rec_name in rec_names:
        coverage = []
        for seed in seeds:
            dir_name = s3.experiment_dir_name(data_dir, env_name, rec_name, seed)
            recommendations = s3.get_and_unserialize(
                bucket_name, dir_name, 'recommendations')
            coverage.append([len(set(rec)) for rec in recommendations])
        coverage = list(compute_stats(np.array(coverage)))
        results.append(coverage)

    plt.figure(figsize=(18, 6))
    for i, rec_name in enumerate(rec_names):
        x_vals = [num_init_ratings + (i * recommendations.shape[1])
                  for i in range(recommendations.shape[0])]
        plt.plot(x_vals, results[i][0], label=rec_name)
        plt.fill_between(x_vals, results[i][2], results[i][1], alpha=0.1)
    plt.xlabel('Number of ratings')
    plt.ylabel('Number of distinct recommended items')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return results


def plot_coverage_vs_ratings(rec_names,
                             len_trial,
                             bucket_name,
                             data_dir,
                             env_name,
                             seeds,
                             num_init_ratings=None):
    """Plot scatterplot of coverage and ratings from data stored in S3.

    Parameters
    ----------
    rec_names : str
        The names of the recommenders for computing novelty.
    len_trial : int
        The number of timesteps per trial.
    bucket_name : str
        The bucket in which the experiment data is saved.
    data_dir: str
        The name of the directory in which the experiment data is saved.
    env_name : str
        The name of the environment for which we are plotting the results.
    seeds : list of int
        The trial seeds across which we are averaging results.
    num_init_ratings : int
        The number of ratings initially available to recommenders. If set to None
        the function will plot with an x-axis based on the timestep.

    """
    rating_stats = plot_ratings_mses_s3(rec_names,
                                        len_trial,
                                        bucket_name,
                                        data_dir,
                                        env_name,
                                        seeds,
                                        num_init_ratings=num_init_ratings,
                                        title=[[''], ['']])
    coverage_stats = plot_coverage_s3(rec_names,
                                      bucket_name,
                                      data_dir,
                                      env_name,
                                      seeds)

    mean_rmses = []
    mean_ratings = []
    coverage = []
    for i in range(len(rating_stats)):
        mean_rmse = rating_stats[rec_names[i]][0][1].mean()
        mean_rating = rating_stats[rec_names[i]][0][0].mean()
        cov = np.mean(coverage_stats[i][0])
        coverage.append(cov)
        mean_rmses.append(mean_rmse)
        mean_ratings.append(mean_rating)

    plt.figure(figsize=[9, 4])
    plt.subplot(1, 2, 1)
    plt.scatter(mean_rmses, coverage)
    for i, rec_name in enumerate(rec_names):
        plt.annotate(rec_names[i], (mean_rmses[i], coverage[i]))
    plt.title('Coverage vs mean RMSE')

    plt.subplot(1, 2, 2)
    plt.scatter(mean_ratings, coverage)
    for i, rec_name in enumerate(rec_names):
        plt.annotate(rec_name, (mean_ratings[i], coverage[i]))
    plt.title('Coverage vs mean ratings')


def plot_ratings_mses_s3(labels,
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
                         title=None):
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
    if title is None:
        title = ['', '']

    def arr_func(ratings, predictions):
        # Setting the predictions for a user/item that has no ratings in the training data to 0.
        predictions[0][predictions[0] > threshold] = 0
        return [ratings[0], (ratings[0] - predictions[0]) ** 2]

    if num_init_ratings is not None and num_users is not None and rating_frequency is not None:
        x_vals = num_init_ratings + num_users * rating_frequency * np.arange(len_trial)
    else:
        x_vals = np.arange(len_trial)

    all_stats = {}
    for label in labels:
        all_stats[label] = compute_stats_s3(bucket_name=bucket_name,
                                            data_dir_name=data_dir_name,
                                            env_name=env_name,
                                            rec_names=[label],
                                            seeds=seeds,
                                            bound_zero=True,
                                            arr_func=arr_func,
                                            load_dense=plot_dense)

    plt.figure(figsize=[9, 4])
    plt.subplot(1, 2, 1)
    for label in labels:
        means, lower_bounds, upper_bounds = all_stats[label]
        means = means[0]
        lower_bounds = lower_bounds[0]
        upper_bounds = upper_bounds[0]
        plt.plot(x_vals, means, label=label)
        plt.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.1)
    plt.xlabel('# ratings')
    plt.ylabel('Mean Rating')
    plt.title(title[0])
    plt.legend()

    plt.subplot(1, 2, 2)
    for label in labels:
        means, lower_bounds, upper_bounds = all_stats[label]
        mse = means[1]
        lower_bounds = lower_bounds[1]
        upper_bounds = upper_bounds[1]
        # Transform the MSE into the RMSE and correct the associated intervals.
        rmse = np.sqrt(mse)
        lower_bounds = np.sqrt(lower_bounds)
        upper_bounds = np.sqrt(upper_bounds)
        plt.plot(x_vals, rmse, label=label)
        plt.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.1)
    plt.xlabel('# ratings')
    plt.ylabel('RMSE')
    plt.title(title[1])
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()
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
    def regret(ratings, _):
        return [np.cumsum(ratings[0] - ratings[1], axis=0)]

    if num_init_ratings is not None and num_users is not None:
        x_vals = num_init_ratings + num_users * np.arange(len_trial)
    else:
        x_vals = np.arange(len_trial)

    plt.figure(figsize=[5, 4])
    for label in labels:
        mean_regrets, lower_bounds, upper_bounds = compute_stats_s3(bucket_name=bucket_name,
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


def compute_stats_s3(bucket_name,
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

    results = s3.compute_across_trials(bucket_name,
                                       data_dir_name,
                                       env_name,
                                       rec_names,
                                       seeds,
                                       get_mean_square_func(arr_func),
                                       load_dense=load_dense)
    means, squares, lengths = zip(*results)
    means = np.average(means, axis=0, weights=lengths)
    squares = np.average(squares, axis=0, weights=lengths)
    variances = squares - means ** 2

    # Apply Bessel's correction.
    num_samples = np.sum(lengths)
    variances = variances * num_samples / (num_samples - 1)

    # Compute the standard error of each sample mean.
    stds = np.sqrt(variances / (num_samples - 1))

    # Compute the 95% confidence intervals using the CLT.
    upper_bounds = means + 2 * stds
    lower_bounds = means - 2 * stds
    if bound_zero:
        lower_bounds = np.maximum(lower_bounds, 0)

    return means, lower_bounds, upper_bounds
