# # Collaborative filtering with private preferences
#
# Model:
#
# - $V$: value
# - $u_i$: user (row) vector
# - $v_j$: item (row) vector
#
# $$V_{ij} = a_i + b_j + u_i v_j^T + x_i y_j^T + \epsilon$$
#
# where $x_i^T y_j$ is the private information known to the user.
#
# At each time $t$, we select a random user $i$ and observe the value corresponding to item
# $$a_{t} = \arg\max_j s_{ijt} + x_i y_j^T$$
# where $s_{ijt}$ is the recommendation score for user $i$, item $j$ at time $t$.
#
# To get initial recommendations, we assume we partially observe the matrix $UV^T$.
#

import collections
import functools as ft
import math
import json
import random

import numpy as np

import scipy as sp
import scipy.linalg


class Schmit(environment.DictEnvironment):
    """
    Implementation of environment with known and unknown user utility, static over time.

    Based on "Human Interaction with Recommendation Systems" by Schmit and Riquelme.

    """

    def __init__(self, num_users, num_items, rating_frequency=0.2,
                 num_init_ratings=0, known_weight=0.98, beta_var=10 ** -5):

        self.num_users = num_users
        self.num_items = num_items

        rank = 10
        sigma = 0.2

        alpha_rank = 10
        nobs_user = int(alpha_rank * rank)

        perc_data = nobs_user / num_items
        print("{} datapoints ({:.1f}% fill / {} observations per user)".format(num_users * nobs_user, 100*perc_data, nobs_user))

        # constants
        item0 = np.random.randn(num_items, 1) / 1.5
        user0 = np.random.randn(num_users, 1) / 3

        # unobserved by agents
        U = np.random.randn(num_users, rank) / np.sqrt(rank)
        V = np.random.randn(num_items, rank) / np.sqrt(rank)
        # observed by agents
        X = np.random.randn(num_users, rank) / np.sqrt(rank)
        Y = np.random.randn(num_items, rank) / np.sqrt(rank)

    def true_score(self, user, item):
        return float(self.item0[item] + self.user0[user] + self.U[user] @ self.V[item].T)

    def value(self, user, item):
        return  float(true_score(user, item) + self.X[user] @ self.Y[item].T + random.gauss(0, sigma))

    def unbiased_value(self, user, item):
        return  true_score(user, item) + random.gauss(0, sigma)

    def sample_user_observations(self, user, score, value, n, test=False):
        # select different items when testing than when training
        mod = 1 if test else 0
        items = sorted(range(num_items), key=lambda i: score(user, i) + X[user] @ Y[i].T, reverse=True)[:(3*n+1)]
        return [(user, item, value(user, item)) for item in items if (user + item) % 2 == mod][:n]

    def sample_data(self, score, value, obs_per_user, test=False):
        return ft.reduce(lambda x, y: x+y,
                         [sample_user_observations(user, score, value, obs_per_user, test)
                          for user in range(num_users)])



        # using perfect scores
        perfect_data = sample_data(true_score, value, nobs_user)
        # user selects data randomly
        random_data = sample_data(lambda u, i: 1000*random.random(), value, nobs_user)
        # scores are 0, user uses preference
        no_score_data = sample_data(lambda u, i: 0, value, nobs_user)

        # unbiased data
        random_unbiased = sample_data(lambda u, i: 1000*random.random(), unbiased_value, nobs_user)
        perfect_unbiased = sample_data(true_score, unbiased_value, nobs_user)

        def avg_value(data, alpha=1):
            n = len(data)
            sum_weights = sum(alpha**i for i in range(n))
            sum_values = sum(alpha**i * value for i, (_, _, value) in enumerate(sorted(data, key=lambda x: -x[2])))
            return sum_values / max(1, sum_weights)



        # group by user
        def groupby(seq, by, vals):
            d = collections.defaultdict(list)
            for item in seq:
                d[by(item)].append(vals(item))

            return d


        def add_constant(A):
            return np.c_[np.ones((A.shape[0], 1)), A]

        quad_loss = lambda x, y: (x - y)**2
        abs_loss = lambda x, y: abs(x - y)

        datasets = [perfect_data, perfect_unbiased, random_data]
        regs = [0.1, 0.5, 1, 3, 5, 10, 25]
