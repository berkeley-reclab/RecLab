"""Default parameters to experimental environments."""
import math

def get_len_trial(ENV_PARAMS):
    """Get length of trial based on the environment parameters.
    """
    if 'num_users' in ENV_PARAMS['params'].keys():
        num_users = ENV_PARAMS['params']['num_users']
        num_items = ENV_PARAMS['params']['num_items']
    elif 'name' in ENV_PARAMS['params']:
        if ENV_PARAMS['params']['name'] == 'ml-100k':                
            num_users = 943
            num_items = 1682
            if 'max_num_users' in ENV_PARAMS['optional_params'].keys():
                num_users = min(num_users, ENV_PARAMS['optional_params']['max_num_users'])
            if 'max_num_items' in ENV_PARAMS['optional_params'].keys():
                num_items = min(num_items, ENV_PARAMS['optional_params']['max_num_items'])
    num_final_ratings = ENV_PARAMS['misc']['num_final_ratings']
    num_init_ratings = ENV_PARAMS['optional_params']['num_init_ratings']
    rating_frequency = ENV_PARAMS['optional_params']['rating_frequency']
    len_trial = math.ceil((num_final_ratings - num_init_ratings) /
                          (num_users * rating_frequency))
    return len_trial

TOPICS_STATIC = {
    'name': 'topics_static',
    'params': {
        'num_topics': 19,
        'num_users': 1000,
        'num_items': 1700,
    },
    'optional_params': {
        'rating_frequency': 0.2,
        'num_init_ratings': 100000,
        'noise': 0.5,
        'topic_change': 0,
        'memory_length': 0,
        'boredom_threshold': 0,
        'boredom_penalty': 0,
    },
    'misc': {
        'num_final_ratings': 200000,
        'sampling': 'uniform',
    },
}
TOPICS_DYNAMIC = {
    'name': 'topics_dynamic',
    'params': {
        'num_topics': 19,
        'num_users': 1000,
        'num_items': 1700,
    },
    'optional_params': {
        'rating_frequency': 0.2,
        'num_init_ratings': 100000,
        'noise': 0.5,
        'topic_change': 0.1,
        'memory_length': 5,
        'boredom_threshold': 2,
        'boredom_penalty': 1,
    },
    'misc': {
        'num_final_ratings': 200000,
        'sampling': 'uniform',
    },
}

LATENT_STATIC = {
    'name': 'latent_static',
    'params': {
        'latent_dim': 100,
        'num_users': 943,
        'num_items': 1682,
    },
    'optional_params': {
        'rating_frequency': 0.2,
        'num_init_ratings': 100000,
        'noise': 0.5,
        'memory_length': 0,
        'boredom_threshold': 0,
        'boredom_penalty': 0,
        'affinity_change': 0,
    },
    'misc': {
        'num_final_ratings': 200000,
        'sampling': 'uniform',
    },
}

LATENT_DYNAMIC = {
    'name': 'latent_dynamic',
    'params': {
        'latent_dim': 100,
        'num_users': 943,
        'num_items': 1682,
    },
    'optional_params': {
        'rating_frequency': 0.2,
        'num_init_ratings': 100000,
        'noise': 0.5,
        'memory_length': 5,
        'boredom_threshold': 0.,
        'boredom_penalty': 2,
        'affinity_change': 0.2,
    },
    'misc': {
        'num_final_ratings': 200000,
        'sampling': 'uniform',
    },
}

SCHMIT = {
    'name': 'schmit',
    'params': {
        'num_users': 1000,
        'num_items': 1700,
    },
    'optional_params': {
        'rating_frequency': 0.2,
        'num_init_ratings': 100000,
        'rank': 10,
        'sigma': 0.2

    },
    'misc': {
        'num_final_ratings': 200000,
    },
}

ENGELHARDT = {
    'name': 'engelhardt',
    'params': {
        'num_topics': 19,
        'num_users': 1000,
        'num_items': 1700,
    },
    'optional_params': {
        'rating_frequency': 0.2,
        'num_init_ratings': 100000,
        'known_weight': 0.98,
        'beta_var': 1e-05,
    },
    'misc': {
        'num_final_ratings': 200000,
    },

}

ML_100K = {
    'name': 'ml_100k',
    'params': {
        'name': 'ml-100k',
    },
    'optional_params': {
        'latent_dim': 100,
        'rating_frequency': 0.2,
        'num_init_ratings': 100000,
        'memory_length': 0,
        'noise': 0.5,
        'boredom_threshold': 0,
        'boredom_penalty': 0,
    },
    'misc': {
        'num_final_ratings': 200000,
        'sampling': 'uniform',
    },
}

ML_100K_LOWDATA = {
    'name': 'ml_100k_lowdata',
    'params': {
        'name': 'ml-100k',
    },
    'optional_params': {
        'latent_dim': 100,
        'rating_frequency': 0.2,
        'num_init_ratings': 20000,
        'memory_length': 0,
        'noise': 0.5,
        'boredom_threshold': 0,
        'boredom_penalty': 0,
    },
    'misc': {
        'num_final_ratings': 120000,
        'sampling': 'uniform',
    },
}

TOPICS_STATIC_SMALL = {
    'name': 'topics_static_small',
    'params': {
        'num_topics': 5,
        'num_users': 100,
        'num_items': 170,
    },
    'optional_params': {
        'rating_frequency': 0.2,
        'num_init_ratings': 1000,
        'noise': 0.5,
        'topic_change': 0,
        'memory_length': 0,
        'boredom_threshold': 0,
        'boredom_penalty': 0,
    },
    'misc': {
        'num_final_ratings': 2000,
        'sampling': 'uniform',
    },
}

TOPICS_DYNAMIC_SMALL = {
    'name': 'topics_dynamic_small',
    'params': {
        'num_topics': 5,
        'num_users': 100,
        'num_items': 170,
    },
    'optional_params': {
        'rating_frequency': 0.2,
        'num_init_ratings': 1000,
        'noise': 0.5,
        'topic_change': 0.1,
        'memory_length': 5,
        'boredom_threshold': 2,
        'boredom_penalty': 1,
    },
    'misc': {
        'num_final_ratings': 2000,
        'sampling': 'uniform',
    },
}

LATENT_STATIC_SMALL = {
    'name': 'latent_static_small',
    'params': {
        'latent_dim': 32,
        'num_users': 100,
        'num_items': 170,
    },
    'optional_params': {
        'rating_frequency': 0.2,
        'num_init_ratings': 1000,
        'noise': 0.5,
        'memory_length': 0,
        'boredom_threshold': 0,
        'boredom_penalty': 0,
    },
    'misc': {
        'num_final_ratings': 2000,
        'sampling': 'uniform',
    },
}

LATENT_DYNAMIC_SMALL = {
    'name': 'latent_dynamic_small',
    'params': {
        'latent_dim': 32,
        'num_users': 100,
        'num_items': 170,
    },
    'optional_params': {
        'rating_frequency': 0.2,
        'num_init_ratings': 1000,
        'noise': 0.5,
        'memory_length': 5,
        'boredom_threshold': 0.,
        'boredom_penalty': 2,
        'affinity_change': 0.2,    },
    'misc': {
        'num_final_ratings': 2000,
        'sampling': 'uniform',
    },
}

SCHMIT_SMALL = {
    'name': 'schmit_small',
    'params': {
        'num_users': 100,
        'num_items': 170,
    },
    'optional_params': {
        'rating_frequency': 0.2,
        'num_init_ratings': 1000,
        'rank': 10,
        'sigma': 0.2

    },
    'misc': {
        'num_final_ratings': 2000,
    },
}

ENGELHARDT_SMALL = {
    'name': 'engelhardt_small',
    'params': {
        'num_topics': 5,
        'num_users': 100,
        'num_items': 170,
    },
    'optional_params': {
        'rating_frequency': 0.2,
        'num_init_ratings': 1000,
        'known_weight': 0.98,
        'beta_var': 1e-05,
    },
    'misc': {
        'num_final_ratings': 2000,
    },
}

FIXED_SMALL = {
    'name': 'fixed_small',
    'params': {
        'num_users': 100,
        'num_items': 170,
    },
    'optional_params': {
        'rating_frequency': 0.2,
        'num_init_ratings': 1000,
    },
    'misc': {
        'num_final_ratings': 2000,
    },
}

ML_100K_SMALL = {
    'name': 'ml_100k_small',
    'params': {
        'name': 'ml-100k',
    },
    'optional_params': {
        'max_num_users': 100,
        'max_num_items': 170,
        'latent_dim': 32,
        'rating_frequency': 0.2,
        'num_init_ratings': 1000,
        'memory_length': 0,
        'noise': 0.5,
        'boredom_threshold': 0,
        'boredom_penalty': 0,
    },
    'misc': {
        'num_final_ratings': 2000,
        'sampling': 'uniform',
    },
}
