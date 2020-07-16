"""A utility module that contains helper functions to run experiments."""
import collections
import copy

import numpy as np
import tqdm.autonotebook

import s3

# The random seed that defines the initial state of each environment.
INIT_SEED = 0


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
                dir_name = s3.experiment_dir_name(data_dir, env_name, rec_name, seed)
                ratings, predictions, dense_ratings, dense_predictions = run_trial(environment,
                                                                                   recommender,
                                                                                   len_trial,
                                                                                   seed,
                                                                                   bucket_name,
                                                                                   dir_name,
                                                                                   overwrite)
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
              bucket_name=None,
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
    bucket_name : str
        The name of the S3 bucket to store the experiment results into. If this is None the results
        will not be saved in S3.
    dir_name : str
        The S3 directory to save the trial results into. Can be None if bucket_name is also None.
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
    if not overwrite and s3.dir_exists(bucket_name, dir_name):
        print('Loading past results from S3 at directory:', dir_name)
        results = s3.load_trial(bucket_name, dir_name)
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
    if bucket_name is not None:
        print('Saving results to S3.')
        s3.save_trial(bucket_name,
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


def rename_duplicates(old_list):
    """Append a number to each element in a list of strings based on the number of duplicates."""
    count = collections.defaultdict(int)
    new_list = []
    for x in old_list:
        new_list.append(x + '_' + str(count[x]))
        count[x] += 1
    return new_list
