import abc

class Recommender(abc.ABC):
    @abc.abstractmethod
    def update_users(self, user_ids, user_data, sparse_update=False):
        """Add or update the information associated with a set of users.

        Parameters
        ----------
        user_ids : int or iterable of int
            The ids of the user(s) to update.
        user_data : sparse matrix
            A matrix where each row represents the features for one user.
        sparse_update : bool
            If set to True a zero entry in user_data will not change that entry if it already
            exists within the recommender, if set to False it will update the entry to zero.
        """
        pass

    @abc.abstractmethod
    def clear_users(self, user_ids):
        """Remove a given set of users.

        Parameters
        ----------
        user_ids : int or iterable of int
            The ids of the user(s) to remove.
        """
        pass

    @abc.abstractmethod
    def update_items(self, item_ids, item_data, sparse_update=False):
        """Add or update the information associated with a set of items.

        Parameters
        ----------
        item_ids : int or iterable of int
            The ids of the item(s) to update.
        item_data : sparse matrix
            A matrix where each row represents the features for one item.
        sparse_update : bool
            If set to True a zero entry in item_data will not change that entry if it already
            exists within the recommender, if set to False it will update the entry to zero.
        """
        pass

    @abc.abstractmethod
    def clear_users(self, item_ids):
        """Remove a given set of users.

        Parameters
        ----------
        user_ids : int or iterable of int
            The ids of the user(s) to remove.
        """
        pass

    @abc.abstractmethod
    def observe_ratings(self, user_ids, item_ids, rating_data, rating_values):
        """Observe new ratings from a set of users on a set of items.

        Parameters
        ----------
        user_ids : int or iterable of int
            The set of users we are observing ratings from. The i-th index corresponds
            to the user making the i-th rating.
        item_ids : int or iterable of int
            The set of item we are observing ratings for. The i-th index corresponds
            to the item being rated in the i-th rating.
        rating_data : sparse matrix
            A matrix where the i-th row represents the features of the i-th rating.
        rating_values : float or iterable of float
            The set set of rating values where the i-th index corresponds to the i-th rating.
        """
        pass

    @abc.abstractmethod
    def predict_scores(self, user_ids, item_ids, rating_data):
        """Predict scores users would give to items.

        Parameters
        ----------
        user_ids : int or iterable of int
            The set of users we are predicting ratings for. The i-th index corresponds
            to the user for which we are predicting the i-th rating.
        item_ids : int or iterable of int
            The set of items we are predicting ratings for. The i-th index corresponds
            to the item for which we are predicting the i-th rating.
        rating_data : sparse matrix
            A matrix where the i-th row represents the features of the i-th rating.
        """
        pass

    @abc.abstractmethod
    def recommend_items(self, user_id, item_ids, rating_data, num_recommendations):
        """Recommend items to a user from a larger set of items.

        Parameters
        ----------
        user_id : int
            The user for which we are recommending items.
        item_ids : iterable of int
            The set of all items from which we can recommend.
        rating_data : sparse matrix
            A matrix where each row corresponds to the rating context in which
            the user would see the item.
        num_recommendations : int
            The number of items to recommend to the user.
        """
        pass

