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

    @property
    @abc.abstractmethod
    def name(self):
        """Get the name of the recommender."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def hyperparameters(self):
        """Get a dict of all the recommender's hyperparameters."""
        raise NotImplementedError

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

    Parameters
    ----------
    strategy_dict : dict, optional
        The item selection strategy to use.
        Valid strategies are:
            {'type': 'greedy'}: chooses the unseen item with largest predicted rating
            {'type': 'eps_greedy', 'eps': 0.x}: with probability 1 - eps chooses the unseen item
            with the largest predicted rating, with probability eps chooses a random unseen item.
            {'type': 'thompson', 'power': x}: picks an item with probability proportional to the
            expected rating raised to power x.

    """

    def __init__(self, **strategy_dict):
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
        # The sampling strategy to use.
        self._strategy_dict = {'type': 'greedy'}
        self.update_strategy(strategy_dict)
        # The cached dense predictions, reset to None each time update is called.
        self._dense_predictions = None
        self._hyperparameters = self._strategy_dict

    @property
    def hyperparameters(self):
        """Get a dict of hyperparameters for this recommender."""
        return self._strategy_dict

    def update_strategy(self, new_strategy):
        """Update the strategy_dict parameter with a new_strategy.

        Parameters
        ----------
        new_strategy : dict
            Contains the exploration strategy parameters.

        """
        if not new_strategy:
            new_strategy = {'type': 'greedy'}

        strategy_type = new_strategy['type']
        if strategy_type == 'eps_greedy':
            eps = new_strategy['eps']
            if (eps < 0) or (eps > 1):
                raise ValueError('eps must be in [0, 1].')
        elif strategy_type == 'thompson':
            power = new_strategy['power']
            if not power.is_integer() or power < 0:
                raise ValueError('power must be a non-negative integer.')
        elif strategy_type != 'greedy':
            raise ValueError('Invalid strategy type.')

        self._strategy_dict = new_strategy

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
        self._dense_predictions = None
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
        self._dense_predictions = None

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
                assert inner_uid < len(self._users)
                assert inner_iid < len(self._items)

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
        # TODO: We need to figure out what to do when the number of items left to recommend
        # runs out.
        for user_id in user_contexts:
            inner_uid = self._outer_to_inner_uid[user_id]
            item_ids = self._ratings[inner_uid].nonzero()[1]
            item_ids = np.setdiff1d(np.arange(len(self._items)), item_ids)
            user_ids = inner_uid * np.ones(len(item_ids), dtype=np.int)
            contexts = len(item_ids) * [user_contexts[user_id]]
            ratings_to_predict += list(zip(user_ids, item_ids, contexts))
            all_item_ids.append(item_ids)

        # Predict the ratings and convert predictions into a list of arrays indexed by user.
        if self._dense_predictions is None:
            all_predictions = self._predict(ratings_to_predict)
        else:
            all_predictions = []
            for user_id, item_id, _ in ratings_to_predict:
                all_predictions.append(self._dense_predictions[user_id, item_id])

        item_lens = map(len, all_item_ids)
        all_predictions = np.split(all_predictions,
                                   list(itertools.accumulate(item_lens)))

        # Pick items according to the strategy, along with their predicted ratings.
        all_recs = []
        all_predicted_ratings = []
        # TODO: Right now items with the same ratings will be sorted in a deterministic order.
        # This probably shouldn't be the case.
        for item_ids, predictions in zip(all_item_ids, all_predictions):
            recs, predicted_ratings = self._select_item(item_ids, predictions,
                                                        num_recommendations)
            # Convert the recommendations to outer item ids.
            all_recs.append([self._inner_to_outer_iid[rec] for rec in recs])
            all_predicted_ratings.append(predicted_ratings)
        return np.array(all_recs), np.array(all_predicted_ratings)

    @property
    def dense_predictions(self):
        """Get the predictions on all user-item pairs.

        This method should be overwritten if there is a more efficient way to compute dense
        predictions than calling _predict on all user-item pairs.
        """
        if self._dense_predictions is None:
            user_item = []
            for i in range(len(self._users)):
                for j in range(len(self._items)):
                    user_item.append((i, j, np.zeros(0)))

            self._dense_predictions = self._predict(user_item)
            self._dense_predictions = self._dense_predictions.reshape((len(self._users),
                                                                       len(self._items)))
        return self._dense_predictions

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

    def _select_item(self, item_ids, predictions, num_recommendations):
        """Select items given a strategy.

        Parameters
        ----------
        item_ids : np.ndarray of int
            ids of the items available for recommendation at this time step
        predictions : np.ndarray
            corresponding predicted ratings for these items
        num_recommendations : int
            number of items to select

        Returns
        -------
        recs : np.ndarray of int
            the indices of the items to be recommended
        predicted_ratings : np.ndarray
            predicted ratings for the selected items

        """
        assert len(item_ids) == len(predictions)
        num_items = len(item_ids)

        strategy_type = self._strategy_dict.get('type')
        if strategy_type == 'greedy':
            selected_indices = np.argsort(predictions)[-num_recommendations:]
        elif strategy_type == 'eps_greedy':
            eps = float(self._strategy_dict.get('eps'))
            num_explore = np.random.binomial(num_recommendations, eps)
            num_exploit = num_recommendations - num_explore
            if num_exploit > 0:
                exploit_indices = np.argsort(predictions)[-num_exploit:]
            else:
                exploit_indices = []
            explore_indices = np.random.choice([x for x in range(0, num_items)
                                                if x not in exploit_indices], num_explore)
            selected_indices = np.concatenate((exploit_indices, explore_indices))
        elif strategy_type == 'thompson':
            power = int(float(self._strategy_dict.get('power')))
            selection_probs = np.power(predictions/sum(predictions), power)
            selection_probs = selection_probs/sum(selection_probs)
            selected_indices = np.random.choice(range(0, num_items),
                                                num_recommendations, p=selection_probs)
        selected_indices = selected_indices.astype('int')
        predicted_ratings = predictions[selected_indices]
        recs = item_ids[selected_indices]
        return recs, predicted_ratings

    @abc.abstractmethod
    def _predict(self, user_item):
        """Predict the ratings of user-item pairs. This internal version assumes inner ids.

        Parameters
        ----------
        user_item : list of tuple
            Each element is a triple where the first element in the tuple is
            the inner user id, the second element is the inner item id and the third element
            is the context in which the item will be rated.

        Returns
        -------
        predictions : np.ndarray
            The rating predictions where predictions[i] is the prediction of the i-th pair.

        """
        raise NotImplementedError
