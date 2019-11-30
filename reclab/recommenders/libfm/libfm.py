"""A wrapper for the LibFM riecommender. See www.libfm.org for implementation details."""
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
            All starting ratings where the key is a double is a double whose first index is the
            id of the user making the rating and the second index is the id of the item being
            rated. The value is a double whose first index is the rating value and the second
            index is a numpy array that represents the context in which the rating was made.

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
            All new ratings where the key is a double is a double whose first index is the
            id of the user making the rating and the second index is the id of the item being
            rated. The value is a double whose first index is the rating value and the second
            index is a numpy array that represents the context in which the rating was made.

        """
        if users is not None:
            self._users.update(users)
        if items is not None:
            self._items.update(items)
        if ratings is not None:
            for (user_id, item_id), (rating, rating_context) in ratings.items():
                assert user_id in self._users
                assert item_id in self._items
                self._rated_items[user_id].add(item_id)
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
        print("Reading in predictions")
        predictions = np.empty(test_inputs.shape[0])
        with open("predictions", "r") as prediction_file:
            for i, line in enumerate(prediction_file):
                predictions[i] = float(line)

        return predictions

    def recommend(self, user_contexts, num_recommendations, strategy="greedy"):
        """Recommend items to users.

        Parameters
        ----------
        user_contexts : ordered dict
            The setting each user is going to be recommended items. The key is the user id and
            the value is the rating features.
        num_recommendations : int
            The number of items to recommend to each user.
        strategy : string, optional
            The type of strategy used to select recommended items, by default "greedy"
            Valid strategies are:
                "greedy" : chooses the unseen item with largest predicted rating
                "eps_greedy" : with probability 1-eps chooses the unseen item with largest
                               predicted rating, with probability eps chooses a random unseen item
                "thompson": picks an item with probability proportional to the expected rating

        Returns
        -------
        recs : np.ndarray of int
            The recommendations made to each user. recs[i] is the array of item ids recommended
            to the i-th user.
        predicted_ratings : np.ndarray
            The predicted ratings of the recommended items. recs[i] is the array of predicted
            ratings for the items recommended to the i-th user.

        """

        # Check that the strategy is of valid type
        valid_strategies = ["greedy", "eps_greedy", "thompson"]
        assert strategy in valid_strategies

        # Format the arrays to be passed to the prediction function. We need to predict all
        # items that have not been rated for each user.
        all_user_ids = []
        all_rating_data = []
        all_item_ids = []
        for i, user_id in enumerate(user_contexts):
            item_ids = np.array([j for j in self._items
                                 if j not in self._rated_items[user_id]])
            all_user_ids.append(user_id * np.ones(len(item_ids), dtype=np.int))
            all_rating_data.append(np.repeat(user_contexts[user_id][np.newaxis, :],
                                             len(item_ids), axis=0))
            all_item_ids.append(item_ids)

        # Predict the ratings and convert predictions into a list of arrays indexed by user.
        all_predictions = self.predict(np.concatenate(all_user_ids),
                                       np.concatenate(all_item_ids),
                                       np.concatenate(all_rating_data))
        item_lens = map(len, all_item_ids)
        all_predictions = np.split(all_predictions,
                                   list(itertools.accumulate(item_lens)))

        # Pick an item according to the strategy, along with their predicted ratings.
        recs = np.zeros((len(user_contexts), num_recommendations), dtype=np.int)
        predicted_ratings = np.zeros(recs.shape)
        for i, (item_ids, predictions) in enumerate(zip(all_item_ids, all_predictions)):
            recs[i], predicted_ratings[i] = select_item(item_ids, predictions,
                                                        num_recommendations, strategy)
        return recs, predicted_ratings


def select_item(item_ids, predictions, num_recommendations, strategy="greedy"):
    """ Helper function that selects items given a strategy

    Parameters
    ----------
    item_ids : np.ndarray of int
        ids of the items available for recommendation at this time step
    predictions : np.ndarray
        corresponding predicted ratings for these items
    num_recommendations : int
        number of items to select
    strategy : str, optional
        item selection strategy, by default "greedy"

    Returns
    -------
    recs: np.ndarray of int
        the indices of the items to be recommended
    predicted_ratings: np.ndarray
        predicted ratings for the selected items
    """

    assert len(item_ids) == len(predictions)
    num_items = len(item_ids)
    if strategy == "greedy":
        selected_indices = np.argpartition(predictions,
                                           num_items - num_recommendations)[-num_recommendations:]
    elif strategy == "eps_greedy":
        eps = 0.1
        num_explore = np.random.binomial(num_recommendations, eps)
        num_exploit = num_recommendations - num_explore
        if num_exploit > 0:
            exploit_indices = np.argpartition(predictions, num_items - num_exploit)[-num_exploit:]
        else:
            exploit_indices = []
        explore_indices = np.random.choice([x for x in range(0, num_items)
                                            if x not in exploit_indices], num_explore)
        selected_indices = np.concatenate((exploit_indices, explore_indices))
    elif strategy == "thompson":
        # artificial parameter to boost the probability of the more likely items
        power = np.ceil(np.log(len(predictions)))
        selection_probs = np.power(predictions/sum(predictions), power)
        selection_probs = selection_probs/sum(selection_probs)
        selected_indices = np.random.choice(range(0, num_items),
                                            num_recommendations, p=selection_probs)

    selected_indices = selected_indices.astype('int')
    predicted_ratings = predictions[selected_indices]
    recs = item_ids[selected_indices]
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
