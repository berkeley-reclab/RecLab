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

    """

    def __init__(self, num_user_features, num_item_features, num_rating_features,
                 max_num_users, max_num_items):
        """Create a LibFM recommender."""
        self._users = {}
        self._max_num_users = max_num_users
        self._items = {}
        self._max_num_items = max_num_items
        self._rated_items = collections.defaultdict(set)
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

    def reset(self, users=None, items=None, ratings=None):
        """Reset the recommender with optional starting user, item, and rating data.

        Parameters
        ----------
        users : dict, optional
            All starting users where the key is the user id while the value is the
            user features.
        items : dict, optional
            All starting items where the key is the user id while the value is the
            item features.
        ratings : np.ndarray, optional
            All starting ratings where ratings[i, 0] is the user id of the i-th rating,
            ratings[i, 1] is the item id of the i-th rating, ratings[i, -1] is the rating
            and the rest of the row represents the rating features.

        """
        self._users = {}
        self._items = {}
        self._rated_items = collections.defaultdict(set)
        self._rating_inputs = scipy.sparse.csr_matrix((0, self._rating_inputs.shape[1]))
        self._rating_outputs = np.empty((0,))
        self.update(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):
        """Update the recommender with new user, item, and rating data.

        Parameters
        ----------
        users : dict, optional
            All new users where the key is the user id while the value is the
            user features.
        items : dict, optional
            All new items where the key is the user id while the value is the
            item features.
        ratings : np.ndarray, optional
            All new ratings where ratings[i, 0] is the user id of the i-th rating,
            ratings[i, 1] is the item id of the i-th rating, ratings[i, -1] is the rating
            and the rest of the row represents the rating features.

        """
        if users is not None:
            self._users.update(users)
        if items is not None:
            self._items.update(items)
        if ratings is not None:
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

    def predict(self, user_ids, item_ids, rating_data):
        """Predict the ratings of user-item pairs.

        Parameters
        ----------
        user_ids : iterable of int
            The list of all user ids for which we wish to predict ratings.
            user_ids[i] is the user id of the i-th pair.
        item_ids : iterable of int
            The list of all item ids for which we wish to predict ratings.
            item_ids[i] is the item id of the i-th pair.
        rating_data : np.ndarray
            The rating features for all the user-item pairs. rating_data[i] is
            the rating features for the i-th pair.

        Returns
        -------
        predictions : np.ndarray
            The rating predictions where predictions[i] is the prediction of the i-th pair.

        """
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
        """Recommend items to users.

        Parameters
        ----------
        user_envs : ordered dict
            The setting each user is going to be recommended items. The key is the user id and
            the value is the rating features.
        num_recommendations : int
            The number of items to recommend to each user.

        Returns
        -------
        recs : np.ndarray of int
            The recommendations made to each user. recs[i] is the array of item ids recommended
            to the i-th user.
        predicted_ratings : np.ndarray
            The predicted ratings of the recommended items. recs[i] is the array of predicted
            ratings for the items recommended to the i-th user.

        """
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
        all_predictions = self.predict(np.concatenate(all_user_ids),
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
    """Write out a train or test file to be used by libfm."""
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
