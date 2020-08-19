"""An implementation of baseline perfect and random recommenders."""
import numpy as np

from . import recommender


class RandomRec(recommender.PredictRecommender):
    """A random recommendation model.

    Parameters
    ----------
    range : tuple
        Upper and lower bounds for the uniformly random predictions.
    seed : int
        The random seed to use for recommendations.

    """

    def __init__(self, rating_range=(0, 5), seed=0):
        """Create a random recommender."""
        self._range = rating_range
        np.random.seed(seed)
        super().__init__()

    @property
    def name(self):  # noqa: D102
        return 'random'

    @property
    def dense_predictions(self):  # noqa: D102
        if self._dense_predictions is None:
            num_users = len(self._users)
            num_items = len(self._items)
            self._dense_predictions = np.random.uniform(low=self._range[0],
                                                        high=self._range[1],
                                                        size=[num_users, num_items])
        return self._dense_predictions

    def _predict(self, user_item):  # noqa: D102
        # Random predictions for all pairs.
        all_predictions = self.dense_predictions
        predictions = []
        for user_id, item_id, _ in user_item:
            predictions.append(all_predictions[user_id, item_id])
        return np.array(predictions)


class PerfectRec(recommender.PredictRecommender):
    """A perfect recommendation model.

    Parameters
    ----------
    dense_rating_function : function
        The function which generates true user ratings.

    """

    def __init__(self, dense_rating_function):
        """Create a perfect recommender."""
        self._dense_rating_function = dense_rating_function
        super().__init__()

    @property
    def name(self):  # noqa: D102
        return 'perfect'

    @property
    def dense_predictions(self):  # noqa: D102
        if self._dense_predictions is None:
            self._dense_predictions = self._dense_rating_function()
        return self._dense_predictions

    def _predict(self, user_item):  # noqa: D102
        # Use provided function to predict for all pairs.
        all_predictions = self.dense_predictions
        predictions = []
        for user_id, item_id, _ in user_item:
            predictions.append(all_predictions[user_id, item_id])
        return np.array(predictions)
