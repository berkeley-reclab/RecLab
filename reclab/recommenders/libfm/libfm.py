import os

import numpy as np
import scipy.sparse

from .. import recommender
from ... import utils


class LibFM(recommender.Recommender):
    def __init__(self, num_user_features, num_item_features, num_rating_features):
        self._users = {}
        self._items = {}
        # Each row of rating_X has the following structure:
        # (user_id, user_features, item_id, item_features, rating_features).
        self._rating_X = scipy.sparse.csr_matrix((0, 1 + num_user_features + 1 +
                                                  num_item_features + num_rating_features))
        # Each row of rating_y consists of the numerical value assigned to that interaction.
        self._rating_y = np.empty((0,))

    def update_users(self, user_ids, user_data, sparse_update=False):
        user_ids = utils.convert_to_iterable(user_ids)
        assert(len(user_ids) == user_data.shape[0])
        for i in range(len(user_ids)):
            if user_ids[i] not in self._users or not sparse_update:
                self._users[user_ids[i]] = user_data[i]
            else:
                nonzero = user_data[i].nonzero()[1]
                self._users[user_ids[i]][0, nonzero] = user_data[i, nonzero]

    def clear_users(self, user_ids):
        user_ids = utils.convert_to_iterable(user_ids)
        for user_id in user_ids:
            self._users.pop(user_id, None)

    def update_items(self, item_ids, item_data, sparse_update=False):
        item_ids = utils.convert_to_iterable(item_ids)
        assert(len(item_ids) == item_data.shape[0])
        for i in range(len(item_ids)):
            if item_ids[i] not in self._items or not sparse_update:
                self._items[item_ids[i]] = item_data[i]
            else:
                nonzero = item_data[i].nonzero()[1]
                self._items[item_ids[i]][0, nonzero] = item_data[i, nonzero]

    def clear_items(self, item_ids):
        item_ids = utils.convert_to_iterable(item_ids)
        for item_id in item_ids:
            self._items.pop(item_id, None)

    def observe_ratings(self, user_ids, item_ids, rating_data, rating_values):
        user_ids = utils.convert_to_iterable(user_ids)
        item_ids = utils.convert_to_iterable(item_ids)
        rating_values = utils.convert_to_iterable(rating_values)
        assert(len(user_ids) == len(item_ids))
        assert(len(item_ids) == len(rating_values))
        assert(len(rating_values) == rating_data.shape[0])
        for i in range(len(user_ids)):
            assert(user_ids[i] in self._users)
            assert(item_ids[i] in self._items)
            user_features = self._users[user_ids[i]]
            item_features = self._items[item_ids[i]]
            new_rating_x = scipy.sparse.hstack((user_ids[i], user_features,
                                                item_ids[i], item_features,
                                                rating_data[i]), format="csr")
            self._rating_X = scipy.sparse.vstack((self._rating_X, new_rating_x), format="csr")
            self._rating_y = np.concatenate((self._rating_y, [rating_values[i]]))

    def predict_scores(self, user_ids, item_ids, rating_data):
        user_ids = utils.convert_to_iterable(user_ids)
        item_ids = utils.convert_to_iterable(item_ids)
        assert(len(user_ids) == len(item_ids))
        assert(len(item_ids) == rating_data.shape[0])

        # Create a test_X array that can be parsed by our output function.
        test_X = scipy.sparse.csr_matrix((0, self._rating_X.shape[1]))
        for i in range(len(user_ids)):
            assert(user_ids[i] in self._users)
            assert(item_ids[i] in self._items)
            user_features = self._users[user_ids[i]]
            item_features = self._items[item_ids[i]]
            new_rating_x = scipy.sparse.hstack((user_ids[i], user_features,
                                                item_ids[i], item_features,
                                                rating_data[i]), format="csr")
            test_X = scipy.sparse.vstack((test_X, new_rating_x), format="csr")

        # Now output both the train and test file.
        self._write_libfm_file("train.libfm", self._rating_X, self._rating_y)
        self._write_libfm_file("test.libfm", test_X, np.zeros(test_X.shape[0]))

        # Run libfm on the train and test files.
        libfm_binary_path = os.path.join(os.path.dirname(__file__), "libfm_lib/bin/libFM")
        os.system("{} -task r -train train.libfm -test test.libfm -dim '1,1,8' -out predictions"
                  .format(libfm_binary_path))

        # Finally read the prediction file back in as a numpy array.
        predictions = np.empty(test_X.shape[0])
        with open("predictions", "r") as f:
            for i, line in enumerate(f):
                predictions[i] = float(line)

        return predictions

    def recommend_items(self, user_id, item_ids, rating_data, num_recommendations):
        assert(len(item_ids) == rating_data.shape[0])
        user_ids = np.ones(len(item_ids), dtype=np.int) * user_id
        predictions = self.predict_scores(user_ids, item_ids, rating_data)
        sorted_indices = np.argsort(predictions)
        return item_ids[sorted_indices[-num_recommendations:]]

    def _write_libfm_file(self, file_path, X, y):
        with open(file_path, "w") as f:
            # TODO: We need to add classification.
            for i in range(X.shape[0]):
                f.write("{} ".format(y[i]))
                indices = X[i].nonzero()[1]
                values = X[i, indices].todense().A1
                index_value_strings = ["{}:{}".format(index, value)
                                       for index, value in zip(indices, values)]
                f.write(" ".join(index_value_strings))
                f.write("\n")

