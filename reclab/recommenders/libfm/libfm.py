"""A wrapper for the LibFM recommender. See www.libfm.org for implementation details."""
import os

import numpy as np
import scipy.sparse

from .. import recommender
from .libfm_lib.bin import pyfm


LIBFM_BINARY_PATH = os.path.join(os.path.dirname(__file__), 'libfm_lib/bin/libFM')


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
        self._train_data = None
        self._num_features = (self._max_num_users + num_user_features + self._max_num_items +
                              num_item_features + num_rating_features)
        self._model = pyfm.PyFM(method="sgd", dim=[1, 1, 8], lr=[0.1])

        # Each row of rating_inputs has the following structure:
        # (user_id, user_features, item_id, item_features, rating_features).
        # Where user_id and item_id are one hot encoded.
        rating_inputs = scipy.sparse.csr_matrix((0, self._num_features))
        # Each row of rating_outputs consists of the numerical value assigned to that interaction.
        rating_outputs = np.empty((0,))
        self._train_data = pyfm.Data(rating_inputs, rating_outputs)

    def reset(self, users=None, items=None, ratings=None):  # noqa: D102
        rating_inputs = scipy.sparse.csr_matrix((0, self._num_features))
        rating_outputs = np.empty((0,))
        self._train_data = pyfm.Data(rating_inputs, rating_outputs)
        super().reset(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        super().update(users, items, ratings)
        if ratings is not None:
            new_rating_inputs = []
            new_rating_outputs = []
            for (user_id, item_id), (rating, rating_context) in ratings.items():
                user_features = self._users[user_id]
                item_features = self._items[item_id]
                one_hot_user_id = scipy.sparse.csr_matrix(([1], ([0], [user_id])),
                                                          shape=(1, self._max_num_users))
                one_hot_item_id = scipy.sparse.csr_matrix(([1], ([0], [item_id])),
                                                          shape=(1, self._max_num_items))
                new_rating_inputs.append(scipy.sparse.hstack((one_hot_user_id, user_features,
                                                              one_hot_item_id, item_features,
                                                              rating_context), format='csr'))
                new_rating_outputs.append(rating)

            new_rating_inputs = scipy.sparse.vstack(new_rating_inputs, format='csr')
            new_rating_outputs = np.array(new_rating_outputs)
            self._train_data.add_rows(new_rating_inputs, new_rating_outputs)

    def _predict(self, user_item):  # noqa: D102
        # Create a test_inputs array that can be parsed by our output function.
        test_inputs = []
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
            test_inputs.append(new_rating_inputs)

        test_inputs = scipy.sparse.vstack(test_inputs, format='csr')
        test_data = pyfm.Data(test_inputs, np.zeros(test_inputs.shape[0]))

        self._model.train(self._train_data)
        predictions = self._model.predict(test_data)

        return predictions

    def model_parameters(self):
        """Train a libfm model and get the resulting model's parameters.

        The factorization machine model predicts a rating by
        r(x) = b_0 + w^T x + v^T V x
        where b_0 is the global bias, w is the weights, and
        V is the pairwise interactions.
        Here, x are the features of the user, item, rating,
        including one-hot encoding of their identity.

        Returns
        -------
        global_bias : float
            global bias term in model
        weights : np.ndarray
            linear term in model (related to user/item biases)
        pairwise_interactions  : np.ndarray
            interaction term in model (related to user/item factors)

        """
        print('Writing libfm file')
        write_libfm_file('train.libfm', self._rating_inputs, self._rating_outputs,
                         self._num_written_ratings)
        self._num_written_ratings = self._rating_inputs.shape[0]
        # Dummy test file
        write_libfm_file('test.libfm', self._rating_inputs[0:1], np.zeros(1))

        print('Running libfm')
        # We use SGD to access save_model (could also use ALS)
        train_command = ('{} -task r -train train.libfm -test test.libfm -method sgd '
                         '-learn_rate 0.01 -regular \'0.04,0.04,0.04\' -dim \'1,1,{}\' '
                         '-verbosity 1 -save_model saved_model'
                         .format(LIBFM_BINARY_PATH, self._latent_dim))
        os.system(train_command)

        # a la https://github.com/jfloff/pywFM/blob/master/pywFM/__init__.py#L238
        global_bias = None
        weights = []
        pairwise_interactions = []
        with open('saved_model', 'rb') as saved_model:
            out_iter = 0
            for _, line in enumerate(saved_model):
                line = line.decode('utf-8')
                if line.startswith('#'):
                    # if out_iter 0 its global bias; if 1, weights; if 2, pairwise interactions
                    if '#global bias W0' in line:
                        out_iter = 0
                    elif '#unary interactions Wj' in line:
                        out_iter = 1
                    elif '#pairwise interactions Vj,f' in line:
                        out_iter = 2
                else:
                    # appends to model parameter according to the flag outer_iter
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
