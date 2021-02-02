"""Implements ModelTuner, a class that automatically tunes a model's hyperparameters."""
import collections
import datetime
import os

import numpy as np
import pandas as pd

import s3


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
        self.bucket_name = bucket_name
        if self.bucket_name is not None:
            if self.data_dir is None:
                raise ValueError('data_dir can not be None when bucket_name is not None.')
            if self.environment_name is None:
                raise ValueError('environment_name can not be None when bucket_name is not None.')
            if self.recommender_name is None:
                raise ValueError('recommender_name can not be None when bucket_name is not None.')

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

                cutoff = 20
                user_true = collections.defaultdict(list)
                user_predicted = collections.defaultdict(list)
                for i in range(len(ratings_to_predict)):
                    user_id = ratings_to_predict[i][0]
                    user_true[user_id].append(true_ratings[i])
                    user_predicted[user_id].append(predicted_ratings[i])
                ndcgs = []
                for user_id in user_true:
                    idcg = get_dcg(get_ranks(user_true[user_id]), user_true[user_id], cutoff=cutoff)
                    dcg = get_dcg(get_ranks(user_predicted[user_id]), user_true[user_id], cutoff=cutoff)
                    ndcg = dcg / idcg
                    ndcgs.append(ndcg)
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
        if self.bucket_name is not None:
            self.s3_save(results, params)
        self.num_evaluations += 1
        return results

    def s3_save(self, results, params):
        """Save the current hyperparameter tuning results to S3."""
        dir_name = os.path.join(self.data_dir, self.environment_name, self.recommender_name,
                                'tuning', 'evaluation_' + str(self.num_evaluations), '')
        if s3.dir_exists(self.bucket_name, dir_name) and not self.overwrite:
            if self.verbose:
                print('Directory:', dir_name, 'already exists. Results will not be saved to S3.')
            return

        if self.verbose:
            print('Saving to S3 directory:', dir_name)
        info = {
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'git branch': s3.git_branch(),
            'git hash': s3.git_hash(),
            'git username': s3.git_username(),
            'recommender': self.recommender_name,
        }
        s3.serialize_and_put(self.bucket_name, dir_name, 'info', info, use_json=True)
        s3.serialize_and_put(self.bucket_name, dir_name, 'params', params, use_json=True)
        s3.put_dataframe(self.bucket_name, dir_name, 'results', results)
