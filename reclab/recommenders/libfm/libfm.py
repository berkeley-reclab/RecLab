"""A wrapper for the LibFM recommender. See www.libfm.org for implementation details."""
import collections
import itertools
import os

import numpy as np
import scipy.sparse


class LibFM():
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
    seed : int
        The seed for the random state of the recommender. Defaults to 0.

    """

    def __init__(self, num_user_features, num_item_features, num_rating_features,
                 max_num_users, max_num_items, seed=0):
        """Create a LibFM recommender."""
        super().__init__
        self._seed = seed
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
        if os.path.exists("train.libfm"):
            os.remove("train.libfm")
        if os.path.exists("test.libfm"):
            os.remove("test.libfm")

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
                                                         rating_context), format="csr")
                self._rating_inputs = scipy.sparse.vstack((self._rating_inputs, new_rating_inputs),
                                                          format="csr")
                self._rating_outputs = np.concatenate((self._rating_outputs, [rating]))

    def _predict(self, user_item):  # noqa: D102
        # Create a test_inputs array that can be parsed by our output function.
        print("Constructing test_inputs")
        test_inputs = scipy.sparse.csr_matrix((0, self._rating_inputs.shape[1]))
        for user_id, item_id, rating in user_item:
            assert user_id in self._users
            assert item_id in self._items
            user_features = self._users[user_id]
            item_features = self._items[item_id]
            one_hot_user_id = scipy.sparse.csr_matrix(([1], ([0], [user_id])),
                                                      shape=(1, self._max_num_users))
            one_hot_item_id = scipy.sparse.csr_matrix(([1], ([0], [item_id])),
                                                      shape=(1, self._max_num_items))
            new_rating_inputs = scipy.sparse.hstack((one_hot_user_id, user_features,
                                                     one_hot_item_id, item_features,
                                                     rating), format="csr")
            test_inputs = scipy.sparse.vstack((test_inputs, new_rating_inputs), format="csr")

        # Now output both the train and test file.
        print("Writing libfm files")
        write_libfm_file("train.libfm", self._rating_inputs, self._rating_outputs,
                         self._num_written_ratings)
        self._num_written_ratings = self._rating_inputs.shape[0]
        write_libfm_file("test.libfm", test_inputs, np.zeros(test_inputs.shape[0]))

        # Run libfm on the train and test files.
        print("Running libfm")
        libfm_binary_path = os.path.join(os.path.dirname(__file__), "libfm_lib/bin/libFM")
        os.system(("{} -task r -train train.libfm -test test.libfm -dim '1,1,8' "
                   "-out predictions -verbosity 1 -seed {}").format(libfm_binary_path, self._seed))

        # Read the prediction file back in as a numpy array.
        print("Reading in predictions")
        predictions = np.empty(test_inputs.shape[0])
        with open("predictions", "r") as prediction_file:
            for i, line in enumerate(prediction_file):
                predictions[i] = float(line)

        return predictions


def write_libfm_file(file_path, inputs, outputs, start_idx=0):
    """Write out a train or test file to be used by libfm."""
    if start_idx == inputs.shape[0]:
        return
    if start_idx == 0:
        write_mode = "w+"
    else:
        write_mode = "a+"
    with open(file_path, write_mode) as out_file:
        for i in range(start_idx, inputs.shape[0]):
            out_file.write("{} ".format(outputs[i]))
            indices = inputs[i].nonzero()[1]
            values = inputs[i, indices].todense().A1
            index_value_strings = ["{}:{}".format(index, value)
                                   for index, value in zip(indices, values)]
            out_file.write(" ".join(index_value_strings) + "\n")
