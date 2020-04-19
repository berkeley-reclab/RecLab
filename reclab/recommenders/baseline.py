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

    def _predict(self, user_item):  # noqa: D102
        # Random predictions for all pairs.
        predictions = np.random.uniform(low=self._range[0],
                                        high=self._range[1],
                                        size=len(user_item))
        return predictions


class PerfectRec(recommender.PredictRecommender):
    """A perfect recommendation model.

    Parameters
    ----------
    rating_function : function
        The function which generates true user ratings.

    """

    def __init__(self, rating_function=0):
        """Create a perfect recommender."""
        self._rating_function = rating_function
        super().__init__()

    @property
    def name(self):  # noqa: D102
        return 'perfect'

    def _predict(self, user_item):  # noqa: D102
        # Use provided functions to predict for all pairs
        predictions = []
        for user_id, item_id, _ in user_item:
            predictions.append(self._rating_function(user_id, item_id))
        return np.array(predictions)
