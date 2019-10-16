import abc

class Environment(abc.ABC):
    def step(self, recommendations):
        """Run one timestep of the environment.

        Parameters
        ----------
        recommendations : iterable
            The recommendations made to each user. The i-th recommendation should correspond to
            the i-th user that was online at this timestep.

        Returns
        -------
        users : iterable
            New users and users whose information got updated this timestep.
        items : iterable
            New items and items whose information got updated this timestep.
        ratings : iterable
            New ratings and ratings whose information got updated this timestep.
        info : dict
            Extra information that can be used for debugging but should not be made accessible to
            the recommender.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def online_users(self):
        """Return the users that need a recommendation at the current timestep.

        Returns
        -------
        users : iterable
            The users that are online.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def all_users(self):
        """Return all users currently in the environment.

        Returns
        -------
        users : iterable
            All users in the environment.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def all_items(self):
        """Return all items currently in the environment.

        Returns
        -------
        items : iterable
            All items in the environment.
        """
        raise NotImplementedError
