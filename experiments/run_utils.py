import numpy as np
import sys, os
sys.path.append('../')

from reclab.recommenders import LibFM
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

class ModelTuner:
    """ This class performs n fold cross validation to
    assess the performance of various model parameters.

    Parameters
    ----------
    x : x
        x

    """

    def __init__(self, data, default_params, n_fold=5, verbose=True):
        """ Create a model tuner. """

        self.users, self.items, self.ratings = data
        self.default_params = default_params
        self.num_users = len(self.users)
        self.num_items = len(self.items)
        self.verbose = verbose
        self._generate_n_folds(n_fold)

    def _generate_n_folds(self, n_fold):
        """ Generate indices for n folds. """

        indices = np.random.permutation(len(self.ratings))
        size_fold = len(self.ratings) // n_fold
        self.train_test_folds = []
        for i in range(n_fold):
            test_ind = indices[i*size_fold:(i+1)*size_fold]
            train_ind = np.append(indices[:i*size_fold], indices[(i+1)*size_fold:])
            self.train_test_folds.append((train_ind, test_ind))

    def evaluate(self, params):
        """ Train and evaluate a model for parameter setting. """

        # constructing model with given parameters
        defaults = {key:self.default_params[key] for key in self.default_params.keys()
                    if key not in params.keys()}
        recommender = LibFM(num_user_features=0, num_item_features=0, num_rating_features=0,
                            max_num_users=self.num_users, max_num_items=self.num_items,
                            **defaults,
                            **params)
        mses = []
        if self.verbose: print("Evaluating:", params)
        for i,fold in enumerate(self.train_test_folds):
            if self.verbose: print("Fold {}/{}, ".format(i+1,len(self.train_test_folds)),
                              end='')
            train_ind, test_ind = fold

            # splitting data dictionaries
            keys = list(self.ratings.keys())
            ratings_test = {key:self.ratings[key] for key in [keys[i] for i in test_ind]}
            ratings_train = {key:self.ratings[key] for key in [keys[i] for i in train_ind]}

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
            if self.verbose: print("mse={}".format(mse))
            mses.append(mse)

        if self.verbose: print("Average MSE:", np.mean(mses))
        return mses

    def evaluate_list(self, param_tag_list):
        """ Train over list of parameters. """

        res_dict = {}
        for tag, params in param_tag_list:
            mses = self.evaluate(params)
            res_dict[tag] = mses
        return res_dict
