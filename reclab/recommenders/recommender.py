"""Defines a set of base classes from which recommenders can inherit.

Recommenders do not need to inherit from any of the classes here to interact with the environments.
However, a recommender that is used in our experiment framework must be a descendent of the
Recommender base class. All the other classes represent specific recommender variants that occur
often enough to be an abstract classes to inherit from.
"""
import abc


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
    def update(self, users=None, items=None, ratings=None): # pylint: disable-unused-argument
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
    """A recommender that makes rating predictions and returns.

    Data is primarily passed around through dicts for any recommenders derived from this class.

    """

    def __init__(self, seed=0):
        self._seed = seed
        self._users = {}
        self._items = {}
        self._ratings = {}
        self._rated_items = collections.defaultdict(set)

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
        self._users = {}
        self._items = {}
        self._ratings = {}
        self._rated_items = collections.defaultdict(set)
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
        if users is not None:
            self._users.update(users)
        if items is not None:
            self._items.update(items)
        if ratings is not None:
            self._ratings.update(ratings)
            for (user_id, item_id), _ in ratings.items():
                assert user_id in self._users
                assert item_id in self._items
                self._rated_items[user_id].add(item_id)

    def recommend(self, users_contexts, num_recommendations):
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
        recs : np.ndarray of int
            The recommendations made to each user. recs[i] is the array of item ids recommended
            to the i-th user.
        predicted_ratings : np.ndarray
            The predicted ratings of the recommended items. recs[i] is the array of predicted
            ratings for the items recommended to the i-th user.

        """
        # Format the arrays to be passed to the prediction function. We need to predict all
        # items that have not been rated for each user.
        ratings_to_predict = []
        all_item_ids = []
        for i, user_id in enumerate(user_contexts):
            item_ids = np.array([j for j in self._items
                                 if j not in self._rated_items[user_id]])
            user_ids = user_id * np.ones(len(item_ids), dtype=np.int)
            contexts = len(item_ids) * [user_contexts[user_id]]
            ratings_to_predict += list(zip(user_ids, item_ids, contexts))
            all_item_ids.append(item_ids)

        # Predict the ratings and convert predictions into a list of arrays indexed by user.
        all_predictions = self.predict(ratings_to_predict)
        item_lens = map(len, all_item_ids)
        all_predictions = np.split(all_predictions,
                                   list(itertools.accumulate(item_lens)))

        # Pick the top predicted items along with their predicted ratings.
        recs = np.zeros((len(user_contexts), num_recommendations), dtype=np.int)
        predicted_ratings = np.zeros(recs.shape)
        for i, (item_ids, predictions) in enumerate(zip(all_item_ids, all_predictions)):
            best_indices = np.argsort(predictions)[-num_recommendations:]
            predicted_ratings[i] = predictions[best_indices]
            recs[i] = item_ids[best_indices]
        return recs, predicted_ratings

    @abc.abstractmethod
    def predict(self, user_ids, item_ids, rating_data):
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
        raise NotImplementedError
