"""Contains make, a function to instantiate a standardized environment from a string."""
from .beta_rank import BetaRank
from .latent_factors import LatentFactorBehavior, DatasetLatentFactor
from .schmit import Schmit
from .topics import Topics

NAMED_ENV_DICT = {
    'topics-static-v1': (
        Topics,
        dict(num_topics=19,
             num_users=1000,
             num_items=1700,
             rating_frequency=0.2,
             num_init_ratings=100000,
             noise=0.5,
             topic_change=0,
             memory_length=0,
             boredom_threshold=0,
             boredom_penalty=0)
    ),
    'topics-dynamic-v1': (
        Topics,
        dict(num_topics=19,
             num_users=1000,
             num_items=1700,
             rating_frequency=0.2,
             num_init_ratings=100000,
             noise=0.5,
             topic_change=0.1,
             memory_length=5,
             boredom_threshold=2,
             boredom_penalty=1)
    ),
    'latent-static-v1': (
        LatentFactorBehavior,
        dict(latent_dim=100,
             num_users=943,
             num_items=1682,
             rating_frequency=0.2,
             num_init_ratings=100000,
             noise=0.5,
             affinity_change=0,
             memory_length=0,
             boredom_threshold=0,
             boredom_penalty=0)
    ),
    'latent-dynamic-v1': (
        LatentFactorBehavior,
        dict(latent_dim=100,
             num_users=943,
             num_items=1682,
             rating_frequency=0.2,
             num_init_ratings=100000,
             noise=0.5,
             affinity_change=0.2,
             memory_length=5,
             boredom_threshold=0,
             boredom_penalty=2)
    ),
    'ml-100k-v1': (
        DatasetLatentFactor,
        dict(name='ml-100k',
             latent_dim=100,
             rating_frequency=0.2,
             num_init_ratings=100000,
             noise=0.5,
             affinity_change=0,
             memory_length=0,
             boredom_threshold=0,
             boredom_penalty=0)
    ),
    'latent-score-v1': (
        Schmit,
        dict(num_users=1000,
             num_items=1700,
             rating_frequency=0.2,
             num_init_ratings=100000,
             rank=10,
             sigma=0.2)
    ),
    'beta-rank-v1': (
        BetaRank,
        dict(num_users=1000,
             num_items=1700,
             num_topics=19,
             rating_frequency=0.2,
             num_init_ratings=100000,
             known_weight=0.98,
             beta_var=1e-05)
    ),
}


def make(name, **kwargs):
    """
    Create an environment by name.

    You may optionally override the arguments for the environment constructor by specifying kwargs.

    Parameters
    ----------
    name : str
        The name of the environment.

    Returns
    ------
    env : Environment
        The constructed environment.

    """
    if name not in NAMED_ENV_DICT:
        raise ValueError('{} is not a valid environment name. '.format(name) +
                         'Valid named environments: {}'.format(NAMED_ENV_DICT.keys()))
    env_class, params = NAMED_ENV_DICT[name]
    params.update(kwargs)
    return env_class(**params)
