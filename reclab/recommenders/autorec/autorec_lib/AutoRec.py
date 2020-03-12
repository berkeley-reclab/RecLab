import tensorflow as tf
import time
import numpy as np
import os
import math

class Autorec():
    def __init__(self,sess,
                      num_users,num_items,
                      R, seen_users, seen_items,
                      hidden_neuron, lambda_value,
                      train_epoch, batch_size, optimizer_method, grad_clip, base_lr,
                      decay_epoch_step, random_seed, display_step, result_path="results"):

        self.sess = sess

        self.num_users = num_users
        self.num_items = num_items

        self.R = R

        self.seen_users = seen_users
        self.seen_items = seen_items

        self.hidden_neuron = hidden_neuron
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.num_batch = int(math.ceil(self.num_users / float(self.batch_size)))

        self.base_lr = base_lr
        self.optimizer_method = optimizer_method
        self.display_step = display_step
        self.random_seed = random_seed

        self.global_step = tf.Variable(0, trainable=False)
        self.decay_epoch_step = decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch
        self.lr = tf.train.exponential_decay(self.base_lr, self.global_step,
                                                   self.decay_step, 0.96, staircase=True)
        self.lambda_value = lambda_value

        self.train_cost_list = []
        self.test_cost_list = []
        self.test_rmse_list = []

        self.result_path = result_path
        self.grad_clip = grad_clip

    def run(self):
        self.prepare_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(self.train_epoch):
            self.train_model(epoch_itr)

    def prepare_model(self):
        self.input_R = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_R")

        V = tf.get_variable(name="V", initializer=tf.truncated_normal(shape=[self.num_items, self.hidden_neuron],
                                         mean=0, stddev=0.03),dtype=tf.float32)
        W = tf.get_variable(name="W", initializer=tf.truncated_normal(shape=[self.hidden_neuron, self.num_items],
                                         mean=0, stddev=0.03),dtype=tf.float32)
        mu = tf.get_variable(name="mu", initializer=tf.zeros(shape=self.hidden_neuron),dtype=tf.float32)
        b = tf.get_variable(name="b", initializer=tf.zeros(shape=self.num_items), dtype=tf.float32)

        pre_Encoder = tf.matmul(self.input_R,V) + mu
        self.Encoder = tf.nn.sigmoid(pre_Encoder)
        pre_Decoder = tf.matmul(self.Encoder,W) + b
        self.Decoder = tf.identity(pre_Decoder)

        pre_rec_cost = self.input_R - self.Decoder
        rec_cost = tf.square(self.l2_norm(pre_rec_cost))
        pre_reg_cost = tf.square(self.l2_norm(W)) + tf.square(self.l2_norm(V))
        reg_cost = self.lambda_value * 0.5 * pre_reg_cost

        self.cost = rec_cost + reg_cost

        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        else:
            raise ValueError("Optimizer Key ERROR")

        if self.grad_clip:
            gvs = optimizer.compute_gradients(self.cost)
            capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        else:
            self.optimizer = optimizer.minimize(self.cost, global_step=self.global_step)

    def train_model(self, itr):
        start_time = time.time()
        random_perm_doc_idx = np.random.permutation(self.num_users)

        batch_cost = 0
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size:]
            elif i < self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size : (i+1) * self.batch_size]


            _, Cost = self.sess.run(
                [self.optimizer, self.cost],
                feed_dict={self.input_R: self.R[batch_set_idx, :]})

            batch_cost = batch_cost + Cost
        self.train_cost_list.append(batch_cost)

        if (itr+1) % self.display_step == 0:
            print ("Training //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(batch_cost),
               "Elapsed time : %d sec" % (time.time() - start_time))

    def predict(self, user_item):
        users = [triple[0] for triple in user_item]
        items = [triple[1] for triple in user_item]

        user_item = zip(users, items)
        Cost, Decoder = self.sess.run(
                [self.cost, self.Decoder],
                feed_dict={self.input_R: self.R})
        user_idx = set(users)
        item_idx = set(items)
        Estimated_R = Decoder.clip(min=1, max=5)
        unseen_user_test_list = list(user_idx - self.seen_users)
        unseen_item_test_list = list(item_idx - self.seen_items)
        for user in unseen_user_test_list:
            for item in unseen_item_test_list:
                 if (user, item) in user_item: # exist in test set
                    Estimated_R[user,item] = 3
        idx = [tuple(users), tuple(items)]
        return np.array(Estimated_R[idx])

    def make_records(self):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        basic_info = self.result_path + "basic_info.txt"
        train_record = self.result_path + "train_record.txt"
        test_record = self.result_path + "test_record.txt"

        with open (train_record,'w') as f:
            f.write(str("Cost:"))
            f.write('\t')
            for itr in range(len(self.train_cost_list)):
                f.write(str(self.train_cost_list[itr]))
                f.write('\t')
            f.write('\n')

        with open (test_record,'w') as g:
            g.write(str("Cost:"))
            g.write('\t')
            for itr in range(len(self.test_cost_list)):
                g.write(str(self.test_cost_list[itr]))
                g.write('\t')
            g.write('\n')

            g.write(str("RMSE:"))
            for itr in range(len(self.test_rmse_list)):
                g.write(str(self.test_rmse_list[itr]))
                g.write('\t')
            g.write('\n')

        with open(basic_info,'w') as h:
            h.write(str(self.args))

    def l2_norm(self,tensor):
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))
