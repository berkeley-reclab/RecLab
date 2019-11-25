"""Baseline recommender that randomly recommend item to user."""
import collections
import itertools
import os
import random

import numpy as np
import scipy.sparse


class Randomrec():
    """The baseline recommendation model which is a random recommender.

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
        """Create a random recommender."""
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
        """Randomly predict the ratings of user-item pairs.

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
        predictions = np.random.uniform(0, 5, item_ids.shape)
        return predictions

    def recommend(self, user_contexts, num_recommendations):
        """Recommend random items to users.

        Parameters
        ----------
        user_contexts : ordered dict
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

        # Pick random items along with their predicted ratings.
        recs = np.zeros((len(user_contexts), num_recommendations), dtype=np.int)
        predicted_ratings = np.zeros(recs.shape)
        for i, (item_ids, predictions) in enumerate(zip(all_item_ids, all_predictions)):
            random_indices = random.sample(range(0, len(predictions)-1), num_recommendations)
            predicted_ratings[i] = predictions[random_indices]
            recs[i] = item_ids[random_indices]
        return recs, predicted_ratings
