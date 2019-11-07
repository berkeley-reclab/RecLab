"""Defines a set of base classes from which environments can inherit.

Environment is the interface all environments must implement. The other classes represent
specific environment variants that occur often enough to be abstract classes to inherit from.
"""
import abc


class Environment(abc.ABC):
    """The interface all environments must implement."""

    @abc.abstractmethod
    def reset(self):
        """Reset the environment to its original state. Must be called before the first step.

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

    @abc.abstractmethod
    def all_ratings(self):
        """Return all ratings that have been made in the environment.

        Returns
        -------
        ratings : iterable
            All ratings in the environment.

        """
        raise NotImplementedError

    def seed(self, seed=None):
        """Set the seed the seed for this environment's random number generator(s)."""
        pass

    def close(self):
        """Perform any necessary cleanup."""
        pass

    def __exit__(self, *args):
        """Perform any necessary cleanup when the object goes out of context."""
        self.close()
        return False
