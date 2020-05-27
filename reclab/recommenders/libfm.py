"""A wrapper for the LibFM recommender. See www.libfm.org for implementation details."""
import numpy as np
import scipy.sparse

import wpyfm
from . import recommender


class LibFM(recommender.PredictRecommender):
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
    use_global_bias : bool
        Whether to use a global bias term.
    use_one_way : bool
        Whether to use one way interactions.
    num_two_way_factors : int
        The number of factors to use for the two way interactions.
    learning_rate : float
        The learning rate for sgd or sgda.
    reg : float
        The regularization across all parameters. Will be overwritten for their respective
        parameters if bias_reg, one_way_reg, or two_way_reg is not None.
    bias_reg : float
        The regularization for the global bias.
    one_way_reg : float
        The regularization for the one-way interactions.
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
                 num_user_features,
                 num_item_features,
                 num_rating_features,
                 max_num_users,
                 max_num_items,
                 method='sgd',
                 use_global_bias=True,
                 use_one_way=True,
                 num_two_way_factors=8,
                 learning_rate=0.1,
                 reg=0.0,
                 bias_reg=None,
                 one_way_reg=None,
                 two_way_reg=None,
                 init_stdev=0.1,
                 num_iter=100,
                 seed=0,
                 **kwargs):
        """Create a LibFM recommender."""
        super().__init__(**kwargs)
        if bias_reg is None:
            bias_reg = reg
        if one_way_reg is None:
            one_way_reg = reg
        if two_way_reg is None:
            two_way_reg = reg
        self._max_num_users = max_num_users
        self._max_num_items = max_num_items
        self._train_data = None
        self._num_features = (self._max_num_users + num_user_features + self._max_num_items +
                              num_item_features + num_rating_features)
        self._model = wpyfm.PyFM(method=method,
                                 dim=(use_global_bias, use_one_way, num_two_way_factors),
                                 lr=learning_rate,
                                 reg=(bias_reg, one_way_reg, two_way_reg),
                                 init_stdev=init_stdev,
                                 num_iter=num_iter,
                                 seed=seed)
        self._hyperparameters.update(locals())
        self._has_xt = method in ('mcmc', 'als')

        # We only want the function arguments so remove class related objects.
        del self._hyperparameters['self']
        del self._hyperparameters['__class__']

        # Each row of rating_inputs has the following structure:
        # (user_id, user_features, item_id, item_features, rating_features).
        # Where user_id and item_id are one hot encoded.
        rating_inputs = scipy.sparse.csr_matrix((0, self._num_features))
        # Each row of rating_outputs consists of the numerical value assigned to that interaction.
        rating_outputs = np.empty((0,))
        self._train_data = wpyfm.Data(rating_inputs, rating_outputs, has_xt=self._has_xt)

    @property
    def name(self):  # noqa: D102
        return 'libfm'

    def reset(self, users=None, items=None, ratings=None):  # noqa: D102
        rating_inputs = scipy.sparse.csr_matrix((0, self._num_features))
        rating_outputs = np.empty((0,))
        self._train_data = wpyfm.Data(rating_inputs, rating_outputs, has_xt=self._has_xt)
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
        test_data = wpyfm.Data(test_inputs, np.zeros(test_inputs.shape[0]), has_xt=self._has_xt)

        if self._has_xt:
            self._model.train(self._train_data, test=test_data)
        else:
            self._model.train(self._train_data)
        predictions = self._model.predict(test_data)

        return predictions

    def model_parameters(self):
        """Train a libfm model and get the resulting model's parameters.

        The degree-2 factorization machine model predicts a rating by

        r(x) = b_0 + w^T x + Ind(j = i) Ind(k = u) V_j^T V_k

        where b_0 is the global bias, w is the weights, and
        V is the pairwise interactions with dimension k * (m+n)
        V_j is the j^th row of V
        x is defined as the concatenation of two one-hot encodings e_i and e_u,
        and w^T x correpond to the user and item biases.

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
