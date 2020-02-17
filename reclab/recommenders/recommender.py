"""Defines a set of base classes from which recommenders can inherit.

Recommenders do not need to inherit from any of the classes here to interact with the environments.
However, a recommender that is used in our experiment framework must be a descendent of the
Recommender base class. All the other classes represent specific recommender variants that occur
often enough to be an abstract classes to inherit from.
"""
import abc
import collections
import itertools

import numpy as np
import scipy


class Recommender(abc.ABC):
    """The interface for recommenders."""

    @abc.abstractmethod
    def reset(self, users=None, items=None, ratings=None):
        """Reset the recommender with optional starting user, item, and rating data.

        Parameters
        ----------
        users : iterable, optional
            The starting users.
        items : iterable, optional
            The starting items.
        ratings : iterable, optional
            The starting ratings and the associated contexts in which each rating was made.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, users=None, items=None, ratings=None):
        """Update the recommender with new user, item, and rating data.

        Parameters
        ----------
        users : iterable, optional
            The new users.
        items : iterable, optional
            The new items.
        ratings : iterable, optional
            The new ratings and the associated context in which each rating was made.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def recommend(self, user_contexts, num_recommendations):
        """Recommend items to users.

        Parameters
        ----------
        user_contexts : iterable
            The setting each user is going to be recommended items in.
        num_recommendations : int
            The number of items to recommend to each user.

        Returns
        -------
        recs : iterable
            The recommendations made to each user. recs[i] represents the recommendations
            made to the i-th user in the user_contexts variable.
        predicted_ratings : iterable or None
            The predicted rating for each item recommended to each user. Where predicted_ratings[i]
            represents the predictions of recommended items on the i-th user in the user_contexts
            variable. Derived classes may simply return None if they do not directly estimate
            ratings when making recommendations.

        """
        raise NotImplementedError


class PredictRecommender(Recommender):
    """A recommender that makes recommendations based on its rating predictions.

    Data is primarily passed around through dicts for any recommenders derived from this class.
    Each user is assumed to have a unique hashable id, likewise for all items. User and item
    features as well as rating contexts are assumed to be dense arrays.

    """

    def __init__(self):
        """Create a new PredictRecommender object."""
        # The features associated with each user.
        self._users = []
        # The features associated with each item.
        self._items = []
        # The matrix of all numerical ratings.
        self._ratings = scipy.sparse.csr_matrix((0, 0))
        # Keeps track of the history of contexts in which a user-item rating was made.
        self._rating_contexts = collections.defaultdict(list)
        # Since outer ids passed to the recommender can be arbitrary hashable objects we
        # use these four variables to keep track of which index (AKA inner ids)
        # correspond to each outer id.
        self._outer_to_inner_uid = {}
        self._inner_to_outer_uid = []
        self._outer_to_inner_iid = {}
        self._inner_to_outer_iid = []

    def reset(self, users=None, items=None, ratings=None):
        """Reset the recommender with optional starting user, item, and rating data.

        Parameters
        ----------
        users : dict, optional
            The starting users where the key is the user id while the value is the
            user features.
        items : dict, optional
            The starting items where the key is the user id while the value is the
            item features.
        ratings : dict, optional
            The starting ratings where the key is a double whose first index is the
            id of the user making the rating and the second index is the id of the item being
            rated. The value is a double whose first index is the rating value and the second
            index is a numpy array that represents the context in which the rating was made.

        """
        self._users = []
        self._items = []
        self._ratings = scipy.sparse.dok_matrix((0, 0))
        self._rating_contexts = collections.defaultdict(list)
        self._outer_to_inner_uid = {}
        self._inner_to_outer_uid = []
        self._outer_to_inner_iid = {}
        self._inner_to_outer_iid = []
        self.update(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):
        """Update the recommender with new user, item, and rating data.

        Parameters
        ----------
        users : dict, optional
            The new users where the key is the user id while the value is the
            user features.
        items : dict, optional
            The new items where the key is the user id while the value is the
            item features.
        ratings : dict, optional
            The new ratings where the key is a double whose first index is the
            id of the user making the rating and the second index is the id of the item being
            rated. The value is a double whose first index is the rating value and the second
            index is a numpy array that represents the context in which the rating was made.

        """
        # Update the user info.
        if users is not None:
            for user_id, features in users.items():
                if user_id not in self._outer_to_inner_uid:
                    self._outer_to_inner_uid[user_id] = len(self._users)
                    self._inner_to_outer_uid.append(user_id)
                    self._ratings.resize((self._ratings.shape[0] + 1, self._ratings.shape[1]))
                    self._users.append(features)
                else:
                    inner_id = self._outer_to_inner_uid[user_id]
                    self._users[inner_id] = features

        # Update the item info.
        if items is not None:
            for item_id, features in items.items():
                if item_id not in self._outer_to_inner_iid:
                    self._outer_to_inner_iid[item_id] = len(self._items)
                    self._inner_to_outer_iid.append(item_id)
                    self._ratings.resize((self._ratings.shape[0], self._ratings.shape[1] + 1))
                    self._items.append(features)
                else:
                    inner_id = self._outer_to_inner_iid[item_id]
                    self._items[inner_id] = features

        # Update the rating info.
        if ratings is not None:
            for (user_id, item_id), (rating, context) in ratings.items():
                inner_uid = self._outer_to_inner_uid[user_id]
                inner_iid = self._outer_to_inner_iid[item_id]
                self._ratings[inner_uid, inner_iid] = rating
                self._rating_contexts[inner_uid, inner_iid].append(context)

    def recommend(self, user_contexts, num_recommendations):
        """Recommend items to users.

        Parameters
        ----------
        user_contexts : ordered dict
            The setting each user is going to be recommended items in. The key is the user id and
            the value is the rating features.
        num_recommendations : int
            The number of items to recommend to each user.

        Returns
        -------
        recs : list of list
            The recommendations made to each user. recs[i] is the list of item ids recommended
            to the i-th user.
        predicted_ratings : list of list
            The predicted ratings of the recommended items. recs[i] is the list of predicted
            ratings for the items recommended to the i-th user.

        """
        # Format the arrays to be passed to the prediction function. We need to predict all
        # items that have not been rated for each user.
        ratings_to_predict = []
        all_item_ids = []
        for user_id in user_contexts:
            inner_uid = self._outer_to_inner_uid[user_id]
            item_ids = np.array([j for j in range(len(self._users))
                                 if self._ratings[inner_uid, j] == 0])
            user_ids = inner_uid * np.ones(len(item_ids), dtype=np.int)
            contexts = len(item_ids) * [user_contexts[user_id]]
            ratings_to_predict += list(zip(user_ids, item_ids, contexts))
            all_item_ids.append(item_ids)

        # Predict the ratings and convert predictions into a list of arrays indexed by user.
        all_predictions = self._predict(ratings_to_predict)
        item_lens = map(len, all_item_ids)
        all_predictions = np.split(all_predictions,
                                   list(itertools.accumulate(item_lens)))

        # Pick the top predicted items along with their predicted ratings.
        all_recs = []
        predicted_ratings = []
        for item_ids, predictions in zip(all_item_ids, all_predictions):
            best_indices = np.argsort(predictions)[-num_recommendations:]
            predicted_ratings.append(predictions[best_indices])
            recs = item_ids[best_indices]
            # Convert the recommendations to outer item ids.
            all_recs.append([self._inner_to_outer_iid[rec] for rec in recs])
        print(all_recs)
        return np.array(all_recs), np.array(predicted_ratings)

    def predict(self, user_item):
        """Predict the ratings of user-item pairs.

        Parameters
        ----------
        user_item : list of tuple
            The list of all user-item pairs along with the rating context.
            Each element is a triple where the first element in the tuple is
            the user id, the second element is the item id and the third element
            is the context in which the item will be rated.

        Returns
        -------
        predictions : np.ndarray
            The rating predictions where predictions[i] is the prediction of the i-th pair.

        """
        inner_user_item = []
        for user_id, item_id, context in user_item:
            inner_uid = self._outer_to_inner_uid[user_id]
            inner_iid = self._outer_to_inner_iid[item_id]
            inner_user_item.append((inner_uid, inner_iid, context))
        return self._predict(inner_user_item)

    @abc.abstractmethod
    def _predict(self, user_item):
        """Predict the ratings of user-item pairs. This internal version assumes inner ids.

        Parameters
        ----------
        user_item : list of tuple
            The list of all user-item pairs along with the rating context.
            Each element is a triple where the first element in the tuple is
            the inner user id, the second element is the inner item id and the third element
            is the context in which the item will be rated.

        Returns
        -------
        predictions : np.ndarray
            The rating predictions where predictions[i] is the prediction of the i-th pair.

        """
        raise NotImplementedError
