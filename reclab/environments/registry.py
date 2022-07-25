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
    'topics-static-v1-small': (
        Topics,
        dict(num_topics=19,
             num_users=100,
             num_items=170,
             rating_frequency=0.2,
             num_init_ratings=5000,
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
    'topics-satiation-v1': (
        Topics,
        dict(num_topics=19,
             num_users=1000,
             num_items=1700,
             rating_frequency=0.2,
             num_init_ratings=100000,
             noise=0.5,
             satiation_factor=3,
             satiation_decay=0.5,
             satiation_noise=0.1)
    ),
    'topics-sensitization-v1': (
        Topics,
        dict(num_topics=19,
             num_users=1000,
             num_items=1700,
             rating_frequency=0.2,
             num_init_ratings=100000,
             noise=0.5,
             satiation_factor=3,
             satiation_decay=(0.1, 0.5),
             satiation_noise=0.1,
             switch_probability=(0.05, 0.2))
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
             latent_dim=0,
             rating_frequency=0.00107,
             num_init_ratings=0,
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
             dimension=19,
             rating_frequency=0.001,
             num_init_ratings=0,
             known_mean=0.98)
    ),
    'beta-rank-lowdata-v1': (
        BetaRank,
        dict(num_users=1000,
             num_items=1700,
             dimension=19,
             rating_frequency=0.001,
             num_init_ratings=0,
             known_mean=0.98)
    ),
    'beta-rank-small-v1': (
        BetaRank,
        dict(num_users=100,
             num_items=170,
             dimension=19,
             rating_frequency=0.01,
             num_init_ratings=0,
             known_mean=0.98)
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
