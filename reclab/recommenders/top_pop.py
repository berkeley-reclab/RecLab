"""An implementation of the top popularity baseline recommender."""
import collections

import numpy as np

from . import recommender


class TopPop(recommender.Recommender):
    """The top popularity recommendation model."""

    def __init__(self):
        """Create a TopPop recommender."""
        self._ratings = collections.defaultdict(dict)
        self._rated_items = collections.defaultdict(set)
        self._ranked_items = []

    @property
    def name(self):  # noqa: D102
        return 'top-pop'

    @property
    def hyperparameters(self):  # noqa: D102
        return {}

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
        self._ratings = collections.defaultdict(dict)
        self._rated_items = collections.defaultdict(set)
        self._ranked_items = []
        self.update(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):  # pylint: disable=unused-argument
        """Update the recommender with new user, item, and rating data.

        Parameters
        ----------
        users : dict, optional
            All new users where the key is the user id while the value is the
            user features.
        items : dict, optional
            All new items where the key is the user id while the value is the
            item features.
        ratings : dict, optional
            All new ratings where the key is a double whose first index is the
            id of the user making the rating and the second index is the id of the item being
            rated. The value is a double whose first index is the rating value and the second
            index is a numpy array that represents the context in which the rating was made.

        """
        # Save all new data.
        item_ids = set(self._ranked_items)
        if items is not None:
            item_ids.update(items)
        if ratings is not None:
            for (user_id, item_id), (rating, _) in ratings.items():
                self._ratings[item_id][user_id] = rating
                self._rated_items[user_id].add(item_id)

        # Compute item averages making sure to take into account items that haven't been rated.
        item_averages = {}
        for item_id in item_ids:
            if item_id in self._ratings:
                item_averages[item_id] = np.mean(list(self._ratings[item_id].values()))
            else:
                item_averages[item_id] = -np.inf

        # Rank all items based on their average.
        self._ranked_items = sorted(item_averages, key=item_averages.get, reverse=True)

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
        recs : np.ndarray of int
            The recommendations made to each user. recs[i] is the array of item ids recommended
            to the i-th user.
        predicted_ratings : np.ndarray
            None since this recommender does not attempt to predict ratings.

        """
        recs = np.zeros((len(user_contexts), num_recommendations), dtype=np.int)
        for i, user_id in enumerate(user_contexts):
            num_items = 0
            for item_id in self._ranked_items:
                if num_items == num_recommendations:
                    break
                if item_id not in self._rated_items[user_id]:
                    recs[i, num_items] = item_id
                    num_items += 1
        return recs, None
