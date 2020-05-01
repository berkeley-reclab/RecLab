"""LLORMA re-imagined"""
import numpy as np
import scipy.sparse

import wpyfm
from . import recommender


class Llorma(recommender.PredictRecommender):
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
    method : str
        The method to learn parameters. Can be one of: 'sgd', 'sgda', or 'mcmc'.
    num_two_way_factors : int
        The number of factors to use for the two way interactions.
    learning_rate : float
        The learning rate for sgd or sgda.
    two_way_reg : float
        The regularization for the two-way interactions.
    init_stdev : float
        Standard deviation for initialization of the 2-way factors.
    num_iter : int
        The number of iterations to train the model for.
    seed : int
        The random seed to use when training the model.

    """

    def __init__(self,
                 max_num_users,
                 max_num_items,
                 num_anchor=8,
                 method='sgd',
                 num_two_way_factors=8,
                 learning_rate=0.1,
                 two_way_reg=0.0,
                 init_stdev=0.1,
                 num_iter=100,
                 seed=0):
        """Create a LLORMA recommender."""
        super().__init__()
        self._max_num_users = max_num_users
        self._max_num_items = max_num_items
        self.num_anchor = num_anchor
        self.rating_data = None
        self.method = method
        self.num_two_way_factors = num_two_way_factors
        self.learning_rate = learning_rate
        self.init_stdev = init_stdev
        self.num_iter = num_iter
        self.seed = seed

    @property
    def name(self):  # noqa: D102
        return 'llorma'

    def reset(self, users=None, items=None, ratings=None):  # noqa: D102
        self.rating_data = None
        super().reset(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        super().update(users, items, ratings)
        if ratings is not None:
            data = []
            row_col = [[], []]
            new_rating_outputs = []
            # TODO: create internal _update function for dealing with inner ids
            for row, ((user_id_outer, item_id_outer),
                      (rating, rating_context)) in enumerate(ratings.items()):
                user_id = self._outer_to_inner_uid[user_id_outer]
                item_id = self._outer_to_inner_iid[item_id_outer]
                user_features = self._users[user_id]
                item_features = self._items[item_id]
                row_col[0].append(row)
                row_col[1].append(user_id)
                data.append(1)
                for i, feature in enumerate(user_features):
                    row_col[0].append(row)
                    row_col[1].append(self._max_num_users + i)
                    data.append(feature)
                row_col[0].append(row)
                row_col[1].append(self._max_num_users + len(user_features) + item_id)
                data.append(1)
                for i, feature in enumerate(item_features):
                    row_col[0].append(row)
                    row_col[1].append(self._max_num_users + len(user_features) +
                                      self._max_num_items + i)
                    data.append(feature)
                for i, feature in enumerate(rating_context):
                    row_col[0].append(row)
                    row_col[1].append(self._max_num_users + len(user_features) +
                                      self._max_num_items + len(item_features) + i)
                    data.append(feature)

                new_rating_outputs.append(rating)

            new_rating_inputs = scipy.sparse.csr_matrix((data, row_col),
                                                        shape=(len(ratings), self._num_features))
            new_rating_outputs = np.array(new_rating_outputs)
            # TODO: We need to account for when the same rating gets added again. Right now
            # this will just add duplicate rows with different ratings.
            self._train_data.add_rows(new_rating_inputs, new_rating_outputs)

    def _predict(self, user_item):  # noqa: D102
        # Create a test_inputs array that can be parsed by our output function.
        test_inputs = []
        data = []
        row_col = [[], []]
        for row, (user_id, item_id, rating_context) in enumerate(user_item):
            user_features = self._users[user_id]
            item_features = self._items[item_id]
            row_col[0].append(row)
            row_col[1].append(user_id)
            data.append(1)
            for i, feature in enumerate(user_features):
                row_col[0].append(row)
                row_col[1].append(self._max_num_users + i)
                data.append(feature)
            row_col[0].append(row)
            row_col[1].append(self._max_num_users + len(user_features) + item_id)
            data.append(1)
            for i, feature in enumerate(item_features):
                row_col[0].append(row)
                row_col[1].append(self._max_num_users + len(user_features) +
                                  self._max_num_items + i)
                data.append(feature)
            for i, feature in enumerate(rating_context):
                row_col[0].append(row)
                row_col[1].append(self._max_num_users + len(user_features) +
                                  self._max_num_items + len(item_features) + i)
                data.append(feature)

        test_inputs = scipy.sparse.csr_matrix((data, row_col),
                                              shape=(len(user_item), self._num_features))
        test_data = wpyfm.Data(test_inputs, np.zeros(test_inputs.shape[0]))

        self._model.train(self._train_data)
        predictions = self._model.predict(test_data)

        return predictions

    def model_parameters(self):
        """Train a libfm model and get the resulting model's parameters.

        The factorization machine model predicts a rating by
        r(x) = b_0 + w^T x + v^T V x
        where b_0 is the global bias, w is the weights, and
        V is the pairwise interactions.
        Here, x are the features of the user, item, rating,
        including one-hot encoding of their identity.

        Returns
        -------
        global_bias : float
            Global bias term in the model.
        weights : np.ndarray
            Linear terms in the model (related to user/item biases).
        pairwise_interactions  : np.ndarray
            Interaction term in the model (related to user/item factors).

        """
        self._model.train(self._train_data)
        return self._model.parameters()

    def hyperparameters(self):
        """Get the hyperparameters associated with this libfm model.

        Returns
        -------
        hyperparameters : dict
            The dict of all hyperparameters.

        """
        return self._hyperparameters
