"""A wrapper for the LibFM recommender. See www.libfm.org for implementation details."""
import itertools
import os

import numpy as np
import scipy.sparse


class LibFM(object):
    """
    """
    def __init__(self, num_user_features, num_item_features, num_rating_features,
                 max_num_users, max_num_items):
        self._users = {}
        self._max_num_users = max_num_users
        self._items = {}
        self._max_num_items = max_num_items
        self._rated_items = {}
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

    def init(self, users, items, ratings):
        self.update(users, items, ratings)

    def update(self, users, items, ratings):
        self._users.update(users)
        for user_id in users:
            if user_id not in self._rated_items:
                self._rated_items[user_id] = set()
        self._items.update(items)
        self.observe_ratings(ratings)

    def clear(self):
        self._users = {}
        self._items = {}
        self._rated_items = {}
        self._rating_inputs = scipy.sparse.csr_matrix((0, self._rating_inputs.shape[1]))
        self._rating_outputs = np.empty((0,))

    def observe_ratings(self, ratings):
        print('Observing ratings with one-hot')
        for i in range(len(ratings)):
            user_id = int(ratings[i, 0])
            item_id = int(ratings[i, 1])
            self._rated_items[user_id].add(item_id)
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
                                                     ratings[i, 2:-1]), format="csr")
            self._rating_inputs = scipy.sparse.vstack((self._rating_inputs, new_rating_inputs),
                                                      format="csr")
            self._rating_outputs = np.concatenate((self._rating_outputs, [ratings[i, -1]]))

    def predict_scores(self, user_ids, item_ids, rating_data):
        assert len(user_ids) == len(item_ids)
        assert len(item_ids) == rating_data.shape[0]

        # Create a test_inputs array that can be parsed by our output function.
        print("Constructing test_inputs")
        test_inputs = scipy.sparse.csr_matrix((0, self._rating_inputs.shape[1]))
        for user_id, item_id, rating in zip(user_ids, item_ids, rating_data):
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
                   "-out predictions -verbosity 1").format(libfm_binary_path))

        # Read the prediction file back in as a numpy array.
        print("Reading in predicitions")
        predictions = np.empty(test_inputs.shape[0])
        with open("predictions", "r") as prediction_file:
            for i, line in enumerate(prediction_file):
                predictions[i] = float(line)

        return predictions

    def recommend(self, user_envs, num_recommendations):
        # Format the arrays to be passed to the prediction function. We need to predict all
        # items that have not been rated for each user.
        all_user_ids = []
        all_rating_data = []
        all_item_ids = []
        for i, user_id in enumerate(user_envs):
            item_ids = np.array([j for j in self._items
                                 if j not in self._rated_items[user_id]])
            all_user_ids.append(user_id * np.ones(len(item_ids), dtype=np.int))
            all_rating_data.append(np.repeat(user_envs[user_id][np.newaxis, :],
                                             len(item_ids), axis=0))
            all_item_ids.append(item_ids)

        # Predict the ratings and convert predictions into a list of arrays indexed by user.
        all_predictions = self.predict_scores(np.concatenate(all_user_ids),
                                              np.concatenate(all_item_ids),
                                              np.concatenate(all_rating_data))
        item_lens = map(len, all_item_ids)
        all_predictions = np.split(all_predictions,
                                   list(itertools.accumulate(item_lens)))

        # Pick the top predicted items along with their predicted ratings.
        recs = np.zeros((len(user_envs), num_recommendations), dtype=np.int)
        predicted_ratings = np.zeros(recs.shape)
        for i, (item_ids, predictions) in enumerate(zip(all_user_ids, all_predictions)):
            best_indices = np.argsort(predictions)[-num_recommendations:]
            predicted_ratings[i] = predictions[best_indices]
            recs[i] = item_ids[best_indices]
        return recs, predicted_ratings


def write_libfm_file(file_path, inputs, outputs, start_idx=0):
    if start_idx == inputs.shape[0]:
        return
    if start_idx == 0:
        write_mode = "w+"
    else:
        write_mode = "a+"
    with open(file_path, write_mode) as out_file:
        # TODO: We need to add classification.
        for i in range(start_idx, inputs.shape[0]):
            out_file.write("{} ".format(outputs[i]))
            indices = inputs[i].nonzero()[1]
            values = inputs[i, indices].todense().A1
            index_value_strings = ["{}:{}".format(index, value)
                                   for index, value in zip(indices, values)]
            out_file.write(" ".join(index_value_strings) + "\n")
