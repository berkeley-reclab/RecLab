"""An implementation of baseline perfect and random recommenders.

See http://glaros.dtc.umn.edu/gkhome/node/774 for details.
"""
import numpy as np

from . import recommender


class RandomRec(recommender.PredictRecommender):
    """A random recommendation model.

    Parameters
    range : tuple
        Upper and lower bounds for the uniformly random predictions.
    seed : int
        The random seed to use for recommendations.

    """

    def __init__(self, rating_range=(0, 5), seed=0):
        """Create a SLIM recommender."""
        self._range = rating_range
        np.random.seed(seed)
        super().__init__()

    def _predict(self, user_item):  # noqa: D102
        # Random predictions for all pairs.
        predictions = []
        for _, _, _ in user_item:
            predictions.append(np.random.uniform(low=self._range[0],
                                                 high=self._range[1]))

        return np.array(predictions)


class PerfectRec(recommender.PredictRecommender):
    """A perfect recommendation model.

    Parameters
    rating_function : function
        The function which generates true user ratings.

    """

    def __init__(self, rating_function=0):
        """Create a SLIM recommender."""
        self._rating_function = rating_function
        # TODO or shoud it be env._rate_item?? is this mutable?
        super().__init__()

    def _predict(self, user_item):  # noqa: D102
        # Use provided functions to predict for all pairs
        predictions = []
        for user_id, item_id, _ in user_item:
            predictions.append(self._rating_function(user_id, item_id))
        return np.array(predictions)
