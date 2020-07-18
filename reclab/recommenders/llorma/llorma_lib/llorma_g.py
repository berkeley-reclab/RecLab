"""
The package for the Global LLORMA recommender.
Code modified from https://github.com/JoonyoungYi/LLORMA-tensorflow
"""
import os
import random
import warnings

import numpy as np
import tensorflow as tf

from .anchor import AnchorManager
from .train_utils import get_train_op, init_latent_mat, init_session

tf.compat.v1.disable_eager_execution()


class Llorma():
    """Local Low Rank Matrix Approximation Model

    Parameters
    ----------
    max_user : int
        Maximum number of users in the environment
    max_item  : int
        Maximum number of items in the environment
    n_anchor : int, optional
        number of anchor-points, by default 10
    pre_rank : int, optional
        latent-dimension of the matrix-factorization model
        used for pre-training, by default 5
    pre_learning_rate : float, optional
        learning rate used to fit the global pre-train model,
        by default 2e-4
    pre_lambda_val : float, optional
        regularization parameter for pre-training,
        by default 10
    pre_train_steps : int, optional
        number of gradient steps used for pretraining,
        by default 100
    rank : int, optional
        latent-dimension of the local models, by default 10
    learning_rate : float, optional
        learning rate used to fit local models, by default 1e-2
    lambda_val : float, optional
        regularization parameter for the local models,
        by default 1e-3
    train_steps : int, optional
        number of train epochs for fitting local models,
        by default 1000
    batch_size : int, optional
        the batch size used when fitting local models,
        by default 1024
    use_cache : bool, optional
        If True use old saved models of the pre-train step,
        by default True
    result_path : str, optional
        directory name where model data will be saved,
        by default 'results'
    kernel_fun : callable, optional
        kernel function used for similarity,
        by_default None
    """
    def __init__(self, max_user, max_item, n_anchor=10, pre_rank=5,
                 pre_learning_rate=2e-4, pre_lambda_val=10,
                 pre_train_steps=100, rank=10, learning_rate=1e-2,
                 lambda_val=1e-3, train_steps=1000, batch_size=1024,
                 use_cache=True, result_path='results', kernel_fun=None):
        """ Initialize a LLORMA recommender
        """
        self.max_user = max_user
        self.max_item = max_item
        self.n_anchor = n_anchor
        self.pre_rank = pre_rank
        self.pre_learning_rate = pre_learning_rate
        self.pre_lambda_val = pre_lambda_val
        self.pre_train_steps = pre_train_steps
        self.rank = rank
        self.learning_rate = learning_rate
        self.lambda_val = lambda_val
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.result_path = result_path
        self.kernel_fun = kernel_fun
        self.user_latent_init = None
        self.item_latent_init = None
        self.batch_manager = None
        self.anchor_manager = None
        self.session = None
        self.model = None
        self.pre_model = None
        self.model = None

    def reset_data(self, train_data, valid_data, test_data):
        """ Reset the data of a recommender by instantiating a
        new BatchManager or modifying the existing one

        Parameters
        ----------
        train_data : Array-like, shape (N_train,3)
            Training data, each row is of the form
            (user_id, item_id, rating)
        valid_data : Array-like, shape (N_valid, 3)
            Validation data, each row is of the form
            (user_id, item_id, rating)
        test_data : Array-like, shape (N_test, 3)
            Test data, each row is of the form
            (user_id, item_idm rating)
        """

        if not self.batch_manager:
            self.batch_manager = BatchManager(train_data, valid_data, test_data)
        else:
            self.batch_manager.update(train_data, valid_data, test_data)

        N_ratings = self.batch_manager.train_data.shape[0]
        if N_ratings < self.n_anchor:
            warnings.warn("The data has fewer ratings than anchor points: {}<{}".format(
                          N_ratings, self.n_anchor))
            self.n_anchor = N_ratings

    def init_pre_model(self):
        """ Initialize TF variables, loss, objective and
        optimizer for the global pre-model
        """
        u_var = tf.compat.v1.placeholder(tf.int64, [None], name='u')
        i_var = tf.compat.v1.placeholder(tf.int64, [None], name='i')
        r_var = tf.compat.v1.placeholder(tf.float64, [None], name='r')

        p_factor = init_latent_mat(self.max_user,
                                   self.pre_rank,
                                   self.batch_manager.mu,
                                   self.batch_manager.std)
        q_factor = init_latent_mat(self.max_item,
                                   self.pre_rank,
                                   self.batch_manager.mu,
                                   self.batch_manager.std)

        p_lookup = tf.nn.embedding_lookup(p_factor, u_var)
        q_lookup = tf.nn.embedding_lookup(q_factor, i_var)
        r_hat = tf.reduce_sum(tf.multiply(p_lookup, q_lookup), 1)

        reg_loss = tf.add_n([tf.reduce_sum(tf.square(p_factor)),
                             tf.reduce_sum(tf.square(q_factor))])
        loss = tf.reduce_sum(tf.square(r_var - r_hat)) + self.pre_lambda_val * reg_loss
        rmse = tf.sqrt(tf.reduce_mean(tf.square(r_var - r_hat)))

        optimizer = tf.compat.v1.train.MomentumOptimizer(self.pre_learning_rate, 0.9)
        train_ops = [
            optimizer.minimize(loss, var_list=[p_factor]),
            optimizer.minimize(loss, var_list=[q_factor])
        ]
        return {
            'u': u_var,
            'i': i_var,
            'r': r_var,
            'train_ops': train_ops,
            'loss': loss,
            'rmse': rmse,
            'p': p_factor,
            'q': q_factor,
        }

    def init_model(self):
        """ Initialize TF variables, loss, objective and
        optimizer for the local models
        """
        u_var = tf.compat.v1.placeholder(tf.int64, [None], name='u')
        i_var = tf.compat.v1.placeholder(tf.int64, [None], name='i')
        r_var = tf.compat.v1.placeholder(tf.float64, [None], name='r')
        k_var = tf.compat.v1.placeholder(tf.float64, [None, self.n_anchor], name='k')
        k_sum = tf.reduce_sum(k_var, axis=1)

        # init weights
        all_p_factors, all_q_factors, r_hats = [], [], []
        for _ in range(self.n_anchor):
            p_factor = init_latent_mat(self.max_user,
                                       self.rank,
                                       self.batch_manager.mu,
                                       self.batch_manager.std)

            q_factor = init_latent_mat(self.max_item,
                                       self.rank,
                                       self.batch_manager.mu,
                                       self.batch_manager.std)
            all_p_factors.append(p_factor)
            all_q_factors.append(q_factor)

            p_lookup = tf.nn.embedding_lookup(p_factor, u_var)
            q_lookup = tf.nn.embedding_lookup(q_factor, i_var)
            r_hat = tf.reduce_sum(tf.multiply(p_lookup, q_lookup), axis=1)
            r_hats.append(r_hat)

        r_hat = tf.reduce_sum(tf.multiply(k_var, tf.stack(r_hats, axis=1)), axis=1)
        r_hat = tf.where(tf.greater(k_sum, 1e-2), r_hat, tf.ones_like(r_hat) * 3)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(r_var - r_hat)))

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate)
        loss = tf.reduce_sum(tf.square(r_hat - r_var)) + self.lambda_val * tf.reduce_sum(
            [tf.reduce_sum(tf.square(p_or_q)) for p_or_q in all_p_factors + all_q_factors])
        train_ops = [get_train_op(optimizer, loss, [p, q])
                     for p, q in zip(all_p_factors, all_q_factors)]
        return {
            'u': u_var,
            'i': i_var,
            'r': r_var,
            'k': k_var,
            'train_ops': train_ops,
            'rmse': rmse,
            'r_hat': r_hat,
        }

    def _get_rmse_pre_model(self, cur_session, pre_model):
        """ Helper method to compute RMSE of the pre-model

        Parameters
        ----------
        cur_session : obj: tf.session
            TensorFlow session to use for computation
        pre_model : Dict-like
            Dictionary of TF variables, train operations
            Typically the output of self.init_pre_model()

        Returns
        -------
        (float, float)
            The validation and test set RMSE
        """
        valid_rmse = cur_session.run(
            pre_model['rmse'],
            feed_dict={
                pre_model['u']: self.batch_manager.valid_data[:, 0],
                pre_model['i']: self.batch_manager.valid_data[:, 1],
                pre_model['r']: self.batch_manager.valid_data[:, 2]
            })

        test_rmse = cur_session.run(
            pre_model['rmse'],
            feed_dict={
                pre_model['u']: self.batch_manager.test_data[:, 0],
                pre_model['i']: self.batch_manager.test_data[:, 1],
                pre_model['r']: self.batch_manager.test_data[:, 2]
            })
        return valid_rmse, test_rmse

    def _get_rmse_model(self, cur_session, model, valid_k, test_k):
        """ Compute the RMSE for the ensamble model of local models

        Parameters
        ----------
        cur_session : obj: tf.session
            TensorFlow session to use for computation
        model : Dict-like
            Dictionary of TF variables, train operations
            Typically the output of self.init_model()
        valid_k : Array-like, shape (N_valid,)
            Kernel weight values for each user-item pair
        test_k : Array-like, shape (N_test,)
            Kernel weight values for each user-item pair

        Returns
        -------
        (float, float)
            The validation and test set RMSE
        """
        valid_rmse = cur_session.run(
            model['rmse'],
            feed_dict={
                model['u']: self.batch_manager.valid_data[:, 0],
                model['i']: self.batch_manager.valid_data[:, 1],
                model['r']: self.batch_manager.valid_data[:, 2],
                model['k']: valid_k,
            })

        test_rmse = cur_session.run(
            model['rmse'],
            feed_dict={
                model['u']: self.batch_manager.test_data[:, 0],
                model['i']: self.batch_manager.test_data[:, 1],
                model['r']: self.batch_manager.test_data[:, 2],
                model['k']: test_k,
            })

        return valid_rmse, test_rmse

    def pre_train(self):  # noqa: R0914
        """Pre-train a Matrix Factorization model for the full data
        """

        # if self.use_cache:
        #     # check if the pre-train factor are already initialized from a previous iteration
        #     if (self.user_latent_init is not None) and (self.item_latent_init is not None):
        #         return

        # if self.pre_model is None:
        #     self.pre_model = self.init_pre_model()

        tf.compat.v1.reset_default_graph()
        self.pre_model = self.init_pre_model()
        pre_model = self.pre_model

        pre_session = tf.compat.v1.Session()
        pre_session.run(tf.compat.v1.global_variables_initializer())

        min_valid_rmse = float('Inf')

        random_model_idx = random.randint(0, 1000000)

        file_path = '{}/pre-model-{}.ckpt'.format(self.result_path, random_model_idx)

        train_data = self.batch_manager.train_data
        u_vec = train_data[:, 0]
        i_vec = train_data[:, 1]
        r_vec = train_data[:, 2]

        #saver = tf.train.Saver()
        for itr in range(self.pre_train_steps):
            for train_op in pre_model['train_ops']:
                pre_session.run((train_op, pre_model['loss'], pre_model['rmse']),
                    feed_dict={pre_model['u']: u_vec,
                               pre_model['i']: i_vec,
                               pre_model['r']: r_vec})
            if (itr+1)%10==0:
                valid_rmse, _ = self._get_rmse_pre_model(pre_session, pre_model)
                print('Pre-train step: {}, train_error:{}'.format(itr+1, valid_rmse))
                # if valid_rmse <= min_valid_rmse:
                #     min_valid_rmse = valid_rmse
                #     min_valid_iter = itr
                # #saver.save(pre_session, file_path)

        #saver.restore(pre_session, file_path)
        p_factor, q_factor = pre_session.run(
            (pre_model['p'], pre_model['q']),
            feed_dict={
                pre_model['u']: self.batch_manager.train_data[:, 0],
                pre_model['i']: self.batch_manager.train_data[:, 1],
                pre_model['r']: self.batch_manager.train_data[:, 2]
            })
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        np.save('{}/pre_train_p.npy'.format(self.result_path), p_factor)
        np.save('{}/pre_train_q.npy'.format(self.result_path), q_factor)

        pre_session.close()

        self.user_latent_init = p_factor
        self.item_latent_init = q_factor

    def train(self):  # noqa: R0914
        """ Train the LLORMA recommender
        """

        self.pre_train()
        #tf.reset_default_graph()
        self.model = self.init_model()
        model = self.model



        self.anchor_manager = AnchorManager(self.n_anchor,
                                            self.batch_manager,
                                            self.user_latent_init,
                                            self.item_latent_init,
                                            self.kernel_fun)
        session = init_session()

        local_models = [
            LocalModel(session, model, anchor_idx, self.anchor_manager, self.batch_manager)
            for anchor_idx in range(self.n_anchor)
        ]

        train_k = _get_local_k(local_models, kind='train')
        valid_k = _get_local_k(local_models, kind='valid')
        test_k = _get_local_k(local_models, kind='test')

        min_valid_rmse = float('Inf')
        min_valid_iter = 0

        train_data = self.batch_manager.train_data

        #saver = tf.train.Saver()
        for itr in range(self.train_steps):
            file_path = '{}/model-{}.ckpt'.format(self.result_path, itr)
            for start_m in range(0, train_data.shape[0], self.batch_size):
                end_m = min(start_m + self.batch_size, train_data.shape[0])
                u_vec = train_data[start_m:end_m, 0]
                i_vec = train_data[start_m:end_m, 1]
                r_vec = train_data[start_m:end_m, 2]
                k_vec = train_k[start_m:end_m, :]
                results = session.run(
                    [model['rmse']] + model['train_ops'],
                    feed_dict={
                        model['u']: u_vec,
                        model['i']: i_vec,
                        model['r']: r_vec,
                        model['k']: k_vec,
                    })


            if (itr+1)%10==0:
                valid_rmse, test_rmse = self._get_rmse_model(session, model,
                                                             valid_k, test_k)
                print("Train step:{}, train error: {}, test error: {}".format(itr+1, test_rmse, valid_rmse))
                # if valid_rmse < min_valid_rmse:
                #     min_valid_rmse = valid_rmse
                #     min_valid_iter = itr
                #     #saver.save(session, file_path)
                #     #saver.restore(session, file_path)

                # if itr >= min_valid_iter + 100:
                #     break



        self.session = session
        return(session, model)

    def predict(self, user_items):
        """Given user-item pairs predict the rating

        Parameters
        ----------
        user_items : Array-like, shape (N, 2)
            Each row is an (user, item) pair

        Returns
        -------
        np.ndarray, shape (N,)
            Predicted ratings
        """

        session = self.session
        model = self.model

        predict_k = np.stack(
            [
                self.anchor_manager.get_k(anchor_idx, user_items)
                for anchor_idx in range(len(self.anchor_manager.anchor_idxs))
            ],
            axis=1)
        predict_k = np.clip(predict_k, 0.0, 1.0)
        predict_k = np.divide(predict_k, np.sum(predict_k, axis=1, keepdims=1))
        predict_k[np.isnan(predict_k)] = 0

        predict_r_hat = session.run(
            model['r_hat'],
            feed_dict={
                model['u']: user_items[:, 0],
                model['i']: user_items[:, 1],
                model['k']: predict_k
            })
        return predict_r_hat


class LocalModel:  # noqa: R0903
    """LocalModel

    Parameters
    ----------
    session : obj: tf.session
        TF session
    model : Dict-like
        Dictionary of TF variables, train operations
        Typically the output of self.init_pre_model()
    anchor_idx : int
        Id of the anchor point for the local model
    anchor_manager : obj: AnchorManager
    batch_manager : obj: BatchManager
    """
    def __init__(self, session, model, anchor_idx, anchor_manager,
                 batch_manager):
        """ Instantiate a local model
        """
        self.session = session
        self.model = model
        self.batch_manager = batch_manager
        self.anchor_idx = anchor_idx
        self.anchor_manager = anchor_manager

        #print('>> update k in anchor_idx [{}].'.format(anchor_idx))
        self.train_k = anchor_manager.get_train_k(anchor_idx)
        self.valid_k = anchor_manager.get_valid_k(anchor_idx)
        self.test_k = anchor_manager.get_test_k(anchor_idx)


def _get_local_k(local_models, kind='train'):
    """Get kernel weights for the local models

    Parameters
    ----------
    local_models : Array-like
        A list of LocalModel objects
    kind : str, optional
        type of data to get kernel weights for,
        possible values are: 'train', 'valid', 'test'
        by default 'train'

    Returns
    -------
    np.ndarray, shape (N_local_models, N_ratings)
        Matrix of kernel weights for each local model
        for each user-item pair in the train/valid/test data
    """
    k = np.stack(
        [
            getattr(local_model, '{}_k'.format(kind))
            for local_model in local_models
        ],
        axis=1)
    k = np.clip(k, 0.0, 1.0)
    k = np.divide(k, np.sum(k, axis=1, keepdims=1))
    k[np.isnan(k)] = 0
    return k


class BatchManager:  # noqa: R0903
    """BatchManager Class to manage the train-valid-test datasets

    Parameters
    ----------
    train_data : Array-like, shape [N_train, 3]
        Each row is of the form (user_id, item_id, rating)
    valid_data : Array-like, shape [N_valid, 3]
        Each row is of the form (user_id, item_id, rating)
    test_data : Array-like, shape [N_test, 3]
        Each row is of the form (user_id, item_id, rating)
    """
    def __init__(self, train_data, valid_data, test_data):
        """Instantiate a BatchManager
        """
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self._set_params()

    def _set_params(self):
        """Private method to set the number of users, number of items,
            mean and standard deviation attributes
        """
        self.n_user = int(
            max(
                np.max(self.train_data[:, 0]),
                np.max(self.valid_data[:, 0]), np.max(self.test_data[:,
                                                                     0]))) + 1
        self.n_item = int(
            max(
                np.max(self.train_data[:, 1]),
                np.max(self.valid_data[:, 1]), np.max(self.test_data[:,
                                                                     1]))) + 1
        self.mu = np.mean(self.train_data[:, 2])
        self.std = np.std(self.train_data[:, 2])

    def update(self, train_data, valid_data=None, test_data=None):
        """ Update the data

        Parameters
        ----------
        train_data : Array-like, shape [N_train, 3]
            Each row is of the form (user_id, item_id, rating)
        valid_data : [Array-like, shape [N_valid, 3], optional
            Each row is of the form (user_id, item_id, rating),
            by default None
        test_data : Array-like, shape [N_test, 3], optional
            Each row is of the form (user_id, item_id, rating)
            by default None
        """
        self.train_data = train_data
        if valid_data is not None:
            self.valid_data = valid_data
        if test_data is not None:
            self.test_data = test_data
        self._set_params()
