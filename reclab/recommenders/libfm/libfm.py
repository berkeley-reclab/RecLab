import os

import numpy as np
import scipy.sparse


class LibFM(object):
    def __init__(self, num_user_features, num_item_features, num_rating_features,
                 max_num_users, max_num_items):
        self._users = {}
        self._max_num_users = max_num_users
        self._items = {}
        self._max_num_items = max_num_items
        self._rated_items = {}
        # Each row of rating_X has the following structure:
        # (user_id, user_features, item_id, item_features, rating_features).
        self._rating_X = scipy.sparse.csr_matrix((0, self._max_num_users + num_user_features +
                                                  self._max_num_items + num_item_features +
                                                  num_rating_features))
        # Each row of rating_y consists of the numerical value assigned to that interaction.
        self._rating_y = np.empty((0,))

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
        self._rating_X = scipy.sparse.csr_matrix((0, 1 + num_user_features + 1 +
                                                  num_item_features + num_rating_features))
        self._rating_y = np.empty((0,))

    def observe_ratings(self, ratings):
        for i in range(len(ratings)):
            user_id = int(ratings[i, 0])
            item_id = int(ratings[i, 1])
            self._rated_items[user_id].add(item_id)
            assert(user_id in self._users)
            assert(item_id in self._items)
            user_features = self._users[user_id]
            item_features = self._items[item_id]
            one_hot_user_id = scipy.sparse.csr_matrix(([1], ([0], [user_id])),
                                                      shape=(1, self._max_num_users))
            one_hot_item_id = scipy.sparse.csr_matrix(([1], ([0], [item_id])),
                                                      shape=(1, self._max_num_items))
            new_rating_x = scipy.sparse.hstack((one_hot_user_id, user_features,
                                                one_hot_item_id, item_features,
                                                ratings[i, 2:-1]), format="csr")
            self._rating_X = scipy.sparse.vstack((self._rating_X, new_rating_x), format="csr")
            self._rating_y = np.concatenate((self._rating_y, [ratings[i, -1]]))

    def predict_scores(self, user_ids, item_ids, rating_data):
        import time
        assert(len(user_ids) == len(item_ids))
        assert(len(item_ids) == rating_data.shape[0])

        # Create a test_X array that can be parsed by our output function.
        test_X = scipy.sparse.csr_matrix((0, self._rating_X.shape[1]))
        for i in range(len(user_ids)):
            assert(user_ids[i] in self._users)
            assert(item_ids[i] in self._items)
            user_features = self._users[user_ids[i]]
            item_features = self._items[item_ids[i]]
            one_hot_user_id = scipy.sparse.csr_matrix(([1], ([0], [user_ids[i]])),
                                                      shape=(1, self._max_num_users))
            one_hot_item_id = scipy.sparse.csr_matrix(([1], ([0], [item_ids[i]])),
                                                      shape=(1, self._max_num_items))
            new_rating_x = scipy.sparse.hstack((one_hot_user_id, user_features,
                                                one_hot_item_id, item_features,
                                                rating_data[i]), format="csr")
            test_X = scipy.sparse.vstack((test_X, new_rating_x), format="csr")

        # Now output both the train and test file.
        s = time.time()
        self._write_libfm_file("train.libfm", self._rating_X, self._rating_y)
        print("1", time.time() - s)
        s = time.time()
        self._write_libfm_file("test.libfm", test_X, np.zeros(test_X.shape[0]))
        print("2", time.time() - s)

        # Run libfm on the train and test files.
        s = time.time()
        libfm_binary_path = os.path.join(os.path.dirname(__file__), "libfm_lib/bin/libFM")
        os.system("{} -task r -train train.libfm -test test.libfm -dim '1,1,8' -out predictions"
                  .format(libfm_binary_path))
        print("3:", time.time() - s)

        # Finally read the prediction file back in as a numpy array.
        predictions = np.empty(test_X.shape[0])
        with open("predictions", "r") as f:
            for i, line in enumerate(f):
                predictions[i] = float(line)

        return predictions

    def recommend(self, user_envs, num_recommendations):
        user_ids = list(user_envs.keys())
        recs = np.zeros((len(user_ids), num_recommendations), dtype=np.int)
        for i, user_id in enumerate(user_ids):
            item_ids = np.array([i for i in self._items.keys() if i not in self._rated_items[user_id]])
            rating_data = np.repeat(user_envs[user_id], len(item_ids), axis=0)
            rating_data = np.zeros((len(item_ids), 0))
            predictions = self.predict_scores(user_id * np.ones(len(item_ids)), item_ids, rating_data)
            sorted_indices = np.argsort(predictions)
            recs[i] = item_ids[sorted_indices[-num_recommendations:]]
        return recs

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

