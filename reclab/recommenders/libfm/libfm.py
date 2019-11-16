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
        self._num_written_ratings = 0
        # Each row of rating_y consists of the numerical value assigned to that interaction.
        self._rating_y = np.empty((0,))
        self._init_libfm_files()

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
        print('Observing ratings with one-hot')
        for i in range(len(ratings)):
            user_id = int(ratings[i, 0])
            item_id = int(ratings[i, 1])
            self._rated_items[user_id].add(item_id)
            assert(user_id in self._users)
            assert(item_id in self._items), [item_id, self._items.keys()]
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
        assert(len(user_ids) == len(item_ids))
        assert(len(item_ids) == rating_data.shape[0])

        # Create a test_X array that can be parsed by our output function.
        print("Constructing test_X")
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
        print("Writing libfm files")
        self._write_libfm_file("train.libfm", self._rating_X, self._rating_y,
                               self._num_written_ratings)
        self._num_written_ratings = self._rating_X.shape[0]
        self._write_libfm_file("test.libfm", test_X, np.zeros(test_X.shape[0]))

        # Run libfm on the train and test files.
        print("Running libfm")
        libfm_binary_path = os.path.join(os.path.dirname(__file__), "libfm_lib/bin/libFM")
        os.system("{} -task r -train train.libfm -test test.libfm -dim '1,1,8' -save_model saved_model -out predictions -verbosity 1"
                  .format(libfm_binary_path))

        # Finally read the prediction file back in as a numpy array.
        print("Reading in predicitions")
        predictions = np.empty(test_X.shape[0])
        with open("predictions", "r") as f:
            for i, line in enumerate(f):
                predictions[i] = float(line)

        return predictions

    def train(self):
        # only training, not predicting.
        print("Writing libfm file")
        self._write_libfm_file("train.libfm", self._rating_X, self._rating_y,
                               self._num_written_ratings)
        self._num_written_ratings = self._rating_X.shape[0]

        # Run libfm on the train and test files.
        print("Running libfm")
        libfm_binary_path = os.path.join(os.path.dirname(__file__), "libfm_lib/bin/libFM")
        os.system("{} -task r -train train.libfm -dim '1,1,8' -verbosity 1 -save_model saved_model"
                  .format(libfm_binary_path))

        # a la https://github.com/jfloff/pywFM/blob/master/pywFM/__init__.py#L238 
        global_bias = None
        weights = []
        pairwise_interactions = []
        with open('saved_model', 'rb') as saved_model:
            # if 0 its global bias; if 1, weights; if 2, pairwise interactions
            out_iter = 0
            for i, line in enumerate(saved_model):
                # checks which line is starting with #
                if line.startswith('#'):
                    if "#global bias W0" in line:
                        out_iter = 0
                    elif "#unary interactions Wj" in line:
                        out_iter = 1
                    elif "#pairwise interactions Vj,f" in line:
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
                        except ValueError as e:
                            pairwise_interactions.append(0.0) #Case: no pairwise interactions used
        pairwise_interactions = np.matrix(pairwise_interactions)
        return global_bias, weights, pairwise_interactions

    def recommend(self, user_envs, num_recommendations):
        user_ids = list(user_envs.keys())
        all_user_ids = np.zeros(0, dtype=np.int)
        all_item_ids = np.zeros(0, dtype=np.int)
        all_rating_data = np.zeros((0, 0), dtype=np.int)
        # TODO: make this a list of arrays instead of a flat list
        num_unseen_items_per_user = np.zeros(len(user_ids), dtype=np.int)
        for i, user_id in enumerate(user_ids):
            item_ids = np.array([j for j in self._items.keys()
                                 if j not in self._rated_items[user_id]])
            rating_data = np.repeat(user_envs[user_id], len(item_ids), axis=0)
            rating_data = np.zeros((len(item_ids), 0))
            all_user_ids = np.concatenate((all_user_ids,
                                           user_id * np.ones(len(item_ids), dtype=np.int)))
            all_item_ids = np.concatenate((all_item_ids, item_ids))
            all_rating_data = np.concatenate((all_rating_data, rating_data))
            num_unseen_items_per_user[i] = len(item_ids)

        predictions = self.predict_scores(all_user_ids, all_item_ids, all_rating_data)
        print('pred shape',predictions.shape)
        print(num_unseen_items_per_user)

        last_idx = 0
        recs = np.zeros((len(user_ids), num_recommendations), dtype=np.int)
        predicted_ratings = []
        for i, length in enumerate(num_unseen_items_per_user):
            item_ids = all_item_ids[last_idx:last_idx + length]
            users_predictions = predictions[last_idx:last_idx + length]
            sorted_indices = np.argsort(users_predictions)
            sorted_ratings = users_predictions[sorted_indices]
            predicted_ratings.append(sorted_ratings[-num_recommendations:])
            recs[i] = item_ids[sorted_indices[-num_recommendations:]]
            last_idx += length
        return recs, np.array(predicted_ratings)

    def _init_libfm_files(self):
        if os.path.exists("train.libfm"):
            os.remove("train.libfm")
        if os.path.exists("test.libfm"):
            os.remove("test.libfm")

    def _write_libfm_file(self, file_path, X, y, start_idx=0):
        if start_idx == X.shape[0]:
            return
        if start_idx == 0:
            write_mode = "w+"
        else:
            write_mode = "a+"
        with open(file_path, write_mode) as f:
            # TODO: We need to add classification.
            for i in range(start_idx, X.shape[0]):
                f.write("{} ".format(y[i]))
                indices = X[i].nonzero()[1]
                values = X[i, indices].todense().A1
                index_value_strings = ["{}:{}".format(index, value)
                                       for index, value in zip(indices, values)]
                f.write(" ".join(index_value_strings) + "\n")

