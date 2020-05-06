TOPICS_STATIC = {
        'name': 'topics_static',
        'params': {
            'num_topics' : 19,
            'num_users' : 1000,
            'num_items' : 1700,
        },
        'optional_params' : {
            'rating_frequency' : 0.2,
            'num_init_ratings' : 100000,
            'noise' : 0.5,
            'topic_change' : 0,
            'memory_length' : 0,
            'boredom_threshold'	: 0,
            'boredom_penalty' : 0,
        },
        'misc' : {
            'num_final_ratings' : 200000,
            'sampling' : 'uniform',
        },
}
TOPICS_DYNAMIC = {
        'name' : 'topics_dynamic',
        'params': {
            'num_topics' : 19,
            'num_users' : 1000,
            'num_items' : 1700,
        },
        'optional_params' : {
            'rating_frequency' : 0.2,
            'num_init_ratings' : 100000,
            'noise' : 0.5,
            'topic_change' : 0.1,
            'memory_length' : 5,
            'boredom_threshold'	: 2,
            'boredom_penalty' : 1,
        },
        'misc' : {
            'num_final_ratings' : 200000,
            'sampling' : 'uniform',
        },
}

LATENT_STATIC = {
        'name' : 'latent_static',
        'params' : {
            'latent_dim' : 100,
            'num_users' : 943,
            'num_items' : 1682,
        },
        'optional_params' : {
            'rating_frequency' : 0.2,
            'num_init_ratings' : 100000,
            'noise' : 0.5,
            'memory_length' : 0,
            'boredom_threshold'	: 0,
            'boredom_penalty' : 0,
        },
        'misc' : {
            'num_final_ratings' : 200000,
            'sampling' : 'uniform',
        },
}

LATENT_DYNAMIC = {
        'name' : 'latent_dynamic',
        'params' : {
            'latent_dim' : 100,
            'num_users' : 943,
            'num_items' : 1682,
        },
        'optional_params' : {
            'rating_frequency' : 0.2,
            'num_init_ratings' : 100000,
            'noise' : 0.5,
            'memory_length' : 5,
            'boredom_threshold'	: 2,
            'boredom_penalty' : 1,
        },
        'misc' : {
            'num_final_ratings' : 200000,
            'sampling' : 'uniform',
        },
}

SCHMIT = {
        'name' : 'schmit',
        'params' : {
            'num_users' : 1000,
            'num_items' : 1700,
        },
        'optional_params' : {
            'rating_frequency' : 0.2,
            'num_init_ratings' : 100000,
            'rank' : 10,
            'sigma' : 0.2

        },
        'misc' : {
            'num_final_ratings' : 200000,
        },
}

ENGELHARDT = {
        'name' : 'engelhardt',
        'params' : {
            'num_topics' : 19,
            'num_users' : 1000,
            'num_items' : 1700,
        },
        'optional_params' : {
            'rating_frequency' : 0.2,
            'num_init_ratings' : 100000,
            'known_weight' : 0.98,
            'beta_var' : 1e-05,
        },
        'misc' : {
            'num_final_ratings' : 200000,
        },

}

ML_100K = {
        'name' : 'ml_100k',
        'params' : {
            'name' : 'ml-100k',
        },
        'optional_params' : {
            'latent_dim' : 100,
            'rating_frequency' : 0.2,
            'num_init_ratings' : 100000,
            'memory_length' : 0,
        },
        'misc' : {
            'num_final_ratings' : 200000,
            'sampling' : 'uniform',
            'noise' : 0.5,
            'topic_change' : 0,
            'memory_length' : 0,
            'boredom_threshold'	: 0,
            'boredom_penalty' : 0,
        },
}