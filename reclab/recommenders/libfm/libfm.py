"""A wrapper for the LibFM recommender. See www.libfm.org for implementation details."""
import os

import numpy as np
import scipy.sparse

from .. import recommender


class LibFM(recommender.PredictRecommender):
    """The libFM recommendation model which is a factorization machine.

    Parameters
    ----------
    num_user_features : int
        The number of features that describe each user.
    num_item_features : int
        The number of features that describe each item.
    num_rating_features : int
        The number of features that describe the context in which each rating occurs.
    max_num_users : int
        The maximum number of users that we will be making predictions for. Note that
        setting this value to be too large will lead to a degradation in performance.
    max_num_items : int
        The maximum number of items that we will be making predictions for. Note that
        setting this value to be too large will lead to a degradation in performance.
    latent_dim : int
        The latent dimension of the factorization model
    seed : int
        The seed for the random state of the recommender. Defaults to 0.

    """

    def __init__(self, num_user_features, num_item_features, num_rating_features,
                 max_num_users, max_num_items, latent_dim=8, seed=0):
        """Create a LibFM recommender."""
        super().__init__()
        self._seed = seed
        self._latent_dim = latent_dim
        self._max_num_users = max_num_users
        self._max_num_items = max_num_items
        # Each row of rating_inputs has the following structure:
        # (user_id, user_features, item_id, item_features, rating_features).
        # Where user_id and item_id are one hot encoded.
        self._rating_inputs = scipy.sparse.csr_matrix((0, self._max_num_users + num_user_features +
                                                       self._max_num_items + num_item_features +
                                                       num_rating_features))
        self._num_written_ratings = 0
        # Each row of rating_outputs consists of the numerical value assigned to that interaction.
        self._rating_outputs = np.empty((0,))

        # Make sure the libfm files are empty.
        if os.path.exists('train.libfm'):
            os.remove('train.libfm')
        if os.path.exists('test.libfm'):
            os.remove('test.libfm')

    def reset(self, users=None, items=None, ratings=None):  # noqa: D102
        self._rating_inputs = scipy.sparse.csr_matrix((0, self._rating_inputs.shape[1]))
        self._rating_outputs = np.empty((0,))
        super().reset(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        super().update(users, items, ratings)
        if ratings is not None:
            for (user_id, item_id), (rating, rating_context) in ratings.items():
                user_features = self._users[user_id]
                item_features = self._items[item_id]
                one_hot_user_id = scipy.sparse.csr_matrix(([1], ([0], [user_id])),
                                                          shape=(1, self._max_num_users))
                one_hot_item_id = scipy.sparse.csr_matrix(([1], ([0], [item_id])),
                                                          shape=(1, self._max_num_items))
                new_rating_inputs = scipy.sparse.hstack((one_hot_user_id, user_features,
                                                         one_hot_item_id, item_features,
                                                         rating_context), format='csr')
                self._rating_inputs = scipy.sparse.vstack((self._rating_inputs, new_rating_inputs),
                                                          format='csr')
                self._rating_outputs = np.concatenate((self._rating_outputs, [rating]))

    def _predict(self, user_item):  # noqa: D102
        # Create a test_inputs array that can be parsed by our output function.
        print('Constructing test_inputs')
        test_inputs = scipy.sparse.csr_matrix((0, self._rating_inputs.shape[1]))
        for user_id, item_id, rating in user_item:
            user_features = self._users[user_id]
            item_features = self._items[item_id]
            one_hot_user_id = scipy.sparse.csr_matrix(([1], ([0], [user_id])),
                                                      shape=(1, self._max_num_users))
            one_hot_item_id = scipy.sparse.csr_matrix(([1], ([0], [item_id])),
                                                      shape=(1, self._max_num_items))
            new_rating_inputs = scipy.sparse.hstack((one_hot_user_id, user_features,
                                                     one_hot_item_id, item_features,
                                                     rating), format='csr')
            test_inputs = scipy.sparse.vstack((test_inputs, new_rating_inputs), format='csr')

        # Now output both the train and test file.
        print('Writing libfm files')
        write_libfm_file('train.libfm', self._rating_inputs, self._rating_outputs,
                         self._num_written_ratings)
        self._num_written_ratings = self._rating_inputs.shape[0]
        write_libfm_file('test.libfm', test_inputs, np.zeros(test_inputs.shape[0]))

        # Run libfm on the train and test files.
        print('Running libfm')
        libfm_binary_path = os.path.join(os.path.dirname(__file__), 'libfm_lib/bin/libFM')
        os.system(('{} -task r -train train.libfm -test test.libfm -dim \'1,1,{}\' '
                   '-out predictions -verbosity 1 -seed {}').format(libfm_binary_path,
                                                                    self._latent_dim, self._seed))

        # Read the prediction file back in as a numpy array.
        print('Reading in predictions')
        predictions = np.empty(test_inputs.shape[0])
        with open('predictions', 'r') as prediction_file:
            for i, line in enumerate(prediction_file):
                predictions[i] = float(line)

        return predictions

    def train(self):
        """Use libfm to train model and reads resulting model.

        Returns
        -------
        global_bias : float
            global bias term in model
        weights : np.ndarray
            linear term in model (user/item biases)
        pairwise_interactions  : np.ndarray
            interaction term in model

        """
        print('Writing libfm file')
        write_libfm_file('train.libfm', self._rating_inputs, self._rating_outputs,
                         self._num_written_ratings)
        self._num_written_ratings = self._rating_inputs.shape[0]
        # Dummy test file
        write_libfm_file('test.libfm', self._rating_inputs[0:1], np.zeros(1))

        print('Running libfm')
        libfm_binary_path = os.path.join(os.path.dirname(__file__), 'libfm_lib/bin/libFM')
        # We use SGD to access save_model (could also use ALS)
        train_command = ('{} -task r -train train.libfm -test test.libfm -method sgd '
                         '-learn_rate 0.01 -regular \'0.04,0.04,0.04\' -dim \'1,1,{}\' '
                         '-verbosity 1 -save_model saved_model'
                         .format(libfm_binary_path, self._latent_dim))
        os.system(train_command)

        # a la https://github.com/jfloff/pywFM/blob/master/pywFM/__init__.py#L238
        global_bias = None
        weights = []
        pairwise_interactions = []
        with open('saved_model', 'rb') as saved_model:
            # if 0 its global bias; if 1, weights; if 2, pairwise interactions
            out_iter = 0
            for _, line in enumerate(saved_model):
                line = line.decode('utf-8')
                # checks which line is starting with #
                if line.startswith('#'):
                    if '#global bias W0' in line:
                        out_iter = 0
                    elif '#unary interactions Wj' in line:
                        out_iter = 1
                    elif '#pairwise interactions Vj,f' in line:
                        out_iter = 2
                else:
                    # check context get in previous step and adds accordingly
                    if out_iter == 0:
                        global_bias = float(line)
                    elif out_iter == 1:
                        weights.append(float(line))
                    elif out_iter == 2:
                        try:
                            pairwise_interactions.append([float(x) for x in line.split(' ')])
                        except ValueError:
                            # Case: no pairwise interactions used
                            pairwise_interactions.append(0.0)
        weights = np.array(weights)
        pairwise_interactions = np.array(pairwise_interactions)

        # Remove the model file
        os.remove('saved_model')

        return global_bias, weights, pairwise_interactions, train_command


def write_libfm_file(file_path, inputs, outputs, start_idx=0):
    """Write out a train or test file to be used by libfm."""
    if start_idx == inputs.shape[0]:
        return
    if start_idx == 0:
        write_mode = 'w+'
    else:
        write_mode = 'a+'
    with open(file_path, write_mode) as out_file:
        for i in range(start_idx, inputs.shape[0]):
            out_file.write('{} '.format(outputs[i]))
            indices = inputs[i].nonzero()[1]
            values = inputs[i, indices].todense().A1
            index_value_strings = ['{}:{}'.format(index, value)
                                   for index, value in zip(indices, values)]
            out_file.write(' '.join(index_value_strings) + '\n')
