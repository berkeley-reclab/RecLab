import os
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.training import optimizer

from .anchor import AnchorManager
from .train_utils import *


class Llorma():
    def __init__(self, batch_manager, n_anchor=10, pre_rank=5,
                 pre_learning_rate=2e-4, pre_lambda_val=10,
                 pre_train_steps=100, rank=10, learning_rate=1e-2,
                 lambda_val=1e-3, train_steps=1000, batch_size=1024,
                 use_cache=True, gpu_memory_frac=0.95, result_path='results'):
        self.batch_manager = batch_manager
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
        self.gpu_memory_frac = gpu_memory_frac
        self.result_path = result_path
        self.user_latent_init = None
        self.item_latent_init = None
        self.anchor_manager = None
        self.session = None
        self.model = None

    def init_pre_model(self):
        u = tf.placeholder(tf.int64, [None], name='u')
        i = tf.placeholder(tf.int64, [None], name='i')
        r = tf.placeholder(tf.float64, [None], name='r')

        p = init_latent_mat(self.batch_manager.n_user,
                            self.pre_rank,
                            self.batch_manager.mu,
                            self.batch_manager.std)
        q = init_latent_mat(self.batch_manager.n_item,
                            self.pre_rank,
                            self.batch_manager.mu,
                            self.batch_manager.std)

        p_lookup = tf.nn.embedding_lookup(p, u)
        q_lookup = tf.nn.embedding_lookup(q, i)
        r_hat = tf.reduce_sum(tf.multiply(p_lookup, q_lookup), 1)

        reg_loss = tf.add_n([tf.reduce_sum(tf.square(p)), tf.reduce_sum(tf.square(q))])
        loss = tf.reduce_sum(tf.square(r - r_hat)) + self.pre_lambda_val * reg_loss
        rmse = tf.sqrt(tf.reduce_mean(tf.square(r - r_hat)))

        optimizer = tf.train.MomentumOptimizer(self.pre_learning_rate, 0.9)
        train_ops = [
            optimizer.minimize(loss, var_list=[p]),
            optimizer.minimize(loss, var_list=[q])
        ]
        return {
            'u': u,
            'i': i,
            'r': r,
            'train_ops': train_ops,
            'loss': loss,
            'rmse': rmse,
            'p': p,
            'q': q,
        }

    def init_model(self):
        u = tf.placeholder(tf.int64, [None], name='u')
        i = tf.placeholder(tf.int64, [None], name='i')
        r = tf.placeholder(tf.float64, [None], name='r')
        k = tf.placeholder(tf.float64, [None, self.n_anchor], name='k')
        k_sum = tf.reduce_sum(k, axis=1)

        # init weights
        ps, qs, losses, r_hats = [], [], [], []
        for anchor_idx in range(self.n_anchor):
            p = init_latent_mat(self.batch_manager.n_user,
                                self.rank,
                                self.batch_manager.mu,
                                self.batch_manager.std)

            q = init_latent_mat(self.batch_manager.n_item,
                                self.rank,
                                self.batch_manager.mu,
                                self.batch_manager.std)
            ps.append(p)
            qs.append(q)

            p_lookup = tf.nn.embedding_lookup(p, u)
            q_lookup = tf.nn.embedding_lookup(q, i)
            r_hat = tf.reduce_sum(tf.multiply(p_lookup, q_lookup), axis=1)
            r_hats.append(r_hat)

        r_hat = tf.reduce_sum(tf.multiply(k, tf.stack(r_hats, axis=1)), axis=1)
        r_hat = tf.where(tf.greater(k_sum, 1e-2), r_hat, tf.ones_like(r_hat) * 3)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(r - r_hat)))

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        loss = tf.reduce_sum(tf.square(r_hat - r)) + self.lambda_val * tf.reduce_sum(
            [tf.reduce_sum(tf.square(p_or_q)) for p_or_q in ps + qs])
        train_ops = [get_train_op(optimizer, loss, [p, q]) for p, q in zip(ps, qs)]
        return {
            'u': u,
            'i': i,
            'r': r,
            'k': k,
            'train_ops': train_ops,
            'rmse': rmse,
            'r_hat': r_hat,
        }

    def _get_rmse_pre_model(self, cur_session, pre_model):
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
        valid_rmse, valid_r_hat = cur_session.run(
            [model['rmse'], model['r_hat']],
            feed_dict={
                model['u']: self.batch_manager.valid_data[:, 0],
                model['i']: self.batch_manager.valid_data[:, 1],
                model['r']: self.batch_manager.valid_data[:, 2],
                model['k']: valid_k,
            })

        test_rmse, test_r_hat = cur_session.run(
            [model['rmse'], model['r_hat']],
            feed_dict={
                model['u']: self.batch_manager.test_data[:, 0],
                model['i']: self.batch_manager.test_data[:, 1],
                model['r']: self.batch_manager.test_data[:, 2],
                model['k']: test_k,
            })

        return valid_rmse, test_rmse

    def pre_train(self):
        if self.use_cache:
            try:
                p = np.load('{}/pre_train_p.npy'.format(self.result_path))
                q = np.load('{}/pre_train_q.npy'.format(self.result_path))
                self.user_latent_init = p
                self.item_latent_init = q
            except FileNotFoundError:
                print('>> There is no cached p and q.')

        pre_model = self.init_pre_model()

        pre_session = tf.Session()
        pre_session.run(tf.global_variables_initializer())

        min_valid_rmse = float("Inf")
        min_valid_iter = 0
        final_test_rmse = float("Inf")

        random_model_idx = random.randint(0, 1000000)

        file_path = "{}/model-{}.ckpt".format(self.result_path, random_model_idx)

        train_data = self.batch_manager.train_data
        u = train_data[:, 0]
        i = train_data[:, 1]
        r = train_data[:, 2]

        saver = tf.train.Saver()
        for iter in range(self.pre_train_steps):
            for train_op in pre_model['train_ops']:
                _, loss, train_rmse = pre_session.run(
                    (train_op, pre_model['loss'], pre_model['rmse']),
                    feed_dict={pre_model['u']: u,
                               pre_model['i']: i,
                               pre_model['r']: r})

            valid_rmse, test_rmse = self._get_rmse_pre_model(pre_session, pre_model)

            if valid_rmse < min_valid_rmse:
                min_valid_rmse = valid_rmse
                min_valid_iter = iter
                final_test_rmse = test_rmse
                saver.save(pre_session, file_path)

            if iter >= min_valid_iter + 100:
                break

            print('>> ITER:',
                  "{:3d}".format(iter), "{:3f}, {:3f} {:3f} / {:3f}".format(
                    train_rmse, valid_rmse, test_rmse, final_test_rmse))

        saver.restore(pre_session, file_path)
        p, q = pre_session.run(
            (pre_model['p'], pre_model['q']),
            feed_dict={
                pre_model['u']: self.batch_manager.train_data[:, 0],
                pre_model['i']: self.batch_manager.train_data[:, 1],
                pre_model['r']: self.batch_manager.train_data[:, 2]
            })
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        np.save('{}/pre_train_p.npy'.format(self.result_path), p)
        np.save('{}/pre_train_q.npy'.format(self.result_path), q)

        pre_session.close()

        self.user_latent_init = p
        self.item_latent_init = q

    def train(self):
        self.pre_train()
        model = self.init_model()

        self.anchor_manager = AnchorManager(self.n_anchor,
                                            self.batch_manager,
                                            self.user_latent_init,
                                            self.item_latent_init)
        session = init_session()

        local_models = [
            LocalModel(session, model, anchor_idx, self.anchor_manager, self.batch_manager)
            for anchor_idx in range(self.n_anchor)
        ]

        train_k = _get_k(local_models, kind='train')
        valid_k = _get_k(local_models, kind='valid')
        test_k = _get_k(local_models, kind='test')

        min_valid_rmse = float("Inf")
        min_valid_iter = 0
        final_test_rmse = float("Inf")
        start_time = time.time()

        batch_rmses = []
        train_data = self.batch_manager.train_data

        for iter in range(self.train_steps):
            for m in range(0, train_data.shape[0], self.batch_size):
                end_m = min(m + self.batch_size, train_data.shape[0])
                u = train_data[m:end_m, 0]
                i = train_data[m:end_m, 1]
                r = train_data[m:end_m, 2]
                k = train_k[m:end_m, :]
                results = session.run(
                    [model['rmse']] + model['train_ops'],
                    feed_dict={
                        model['u']: u,
                        model['i']: i,
                        model['r']: r,
                        model['k']: k,
                    })
                batch_rmses.append(results[0])

                if m % (self.batch_size * 100) == 0:
                    print('  - ', results[:1])

            if iter % 1 == 0:
                valid_rmse, test_rmse = self._get_rmse_model(session, model,
                                                             valid_k, test_k)
                if valid_rmse < min_valid_rmse:
                    min_valid_rmse = valid_rmse
                    min_valid_iter = iter
                    final_test_rmse = test_rmse

                batch_rmse = sum(batch_rmses) / len(batch_rmses)
                batch_rmses = []
                print('  - ITER{:4d}:'.format(iter),
                      "{:.5f}, {:.5f} {:.5f} / {:.5f}".format(
                        batch_rmse, valid_rmse, test_rmse, final_test_rmse))

        self.session = session
        self.model = model
        return(session, model)

    def predict(self, user_items):
        session = self.session
        model = self.model
        predict_k = np.stack(
            [
                self.anchor_manager._get_k(anchor_idx, user_items)
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


class LocalModel:
    def __init__(self, session, model, anchor_idx, anchor_manager,
                 batch_manager):
        self.session = session
        self.model = model
        self.batch_manager = batch_manager
        self.anchor_idx = anchor_idx
        self.anchor_manager = anchor_manager

        print('>> update k in anchor_idx [{}].'.format(anchor_idx))
        self.train_k = anchor_manager.get_train_k(anchor_idx)
        self.valid_k = anchor_manager.get_valid_k(anchor_idx)
        self.test_k = anchor_manager.get_test_k(anchor_idx)


def _get_k(local_models, kind='train'):
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


class BatchManager:
    def __init__(self, train_data, valid_data, test_data):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self._set_params()

    def _set_params(self):
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


    def update_self(self, train_data, valid_data=None, test_data=None):
        self.train_data = train_data
        if valid_data:
            self.valid_data = valid_data
        if test_data:
            self.test_data = test_data
