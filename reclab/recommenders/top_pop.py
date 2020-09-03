"""An implementation of the top popularity baseline recommender."""

import numpy as np
import scipy.sparse

from . import recommender


# TODO: add flag to allow this to also be based on number of times rated.
class TopPop(recommender.PredictRecommender):
    """The top popularity recommendation model based on ratings."""

    @property
    def name(self):  # noqa: D102
        return 'top-pop'

    @property
    def dense_predictions(self):  # noqa: D102
        if self._dense_predictions is None:
            item_vector = self._average_item_ratings()
            self._dense_predictions = np.vstack([item_vector] * self._ratings.shape[0])
        return self._dense_predictions

    def _average_item_ratings(self):
        # Compute average rating of each item
        row, col = self._ratings.nonzero()
        data = np.ones(len(row))
        binary_ratings = scipy.sparse.csr_matrix((data, (row, col)), shape=self._ratings.shape)

        summed_item_ratings = self._ratings.sum(0)
        num_times_rated = binary_ratings.sum(0)

        item_vector = -np.inf * np.ones(num_times_rated.shape)
        idx_rated = np.where(num_times_rated > 0)
        item_vector[idx_rated] = summed_item_ratings[idx_rated] / num_times_rated[idx_rated]

        return item_vector.flatten()

    def _predict(self, user_item):  # noqa: D102
        # Predict on all user-item pairs.
        average_item_ratings = self._average_item_ratings()
        predictions = []
        for _, item_id, _ in user_item:
            predictions.append(average_item_ratings[item_id])

        return np.array(predictions)
