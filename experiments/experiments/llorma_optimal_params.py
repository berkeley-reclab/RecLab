OPT_TOPICS_SMALL = {
                'batch_size': 128,
                'lambda_val': 0.01,
                'learning_rate': 0.02,
                'n_anchor': 10,
                'pre_lambda_val': 0.1,
                'pre_learning_rate': 0.001,
                'pre_rank': 5,
                'pre_train_steps': 20,
                'rank': 10,
                'result_path': 'results_static_small',
                'train_steps': 100,
                'use_cache': False}

OPT_TOPICS = {
                'result_path': 'results_static',
                'n_anchor': 10,
                'pre_rank': 10,
                'pre_learning_rate': 3e-4,
                'pre_lambda_val': 0.01,
                'pre_train_steps': 70,
                'rank': 20,
                'learning_rate': 2e-2,
                'lambda_val': 1e-4,
                'train_steps': 50,
                'batch_size': 1000,
                'use_cache': False}

OPT_LATENT= {'result_path': 'results_dynamic',
            'n_anchor': 10,
            'pre_rank': 10,
            'pre_learning_rate': 2e-4,
            'pre_lambda_val': 0.01,
            'pre_train_steps': 50,
            'rank': 20,
            'train_steps': 70,
            'batch_size': 1000,
            'use_cache': False}

LEARNING_RATE = [5e-3, 1e-2, 2e-2, 4e-2]
LAMBDA_VAL = [1e-4, 2e-4, 5e-4, 1e-3]
