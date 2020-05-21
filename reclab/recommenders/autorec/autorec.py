"""Pytorch implementation of AutoRec recommender."""

import math
import numpy as np
import torch

from .autorec_lib import autorec
from .. import recommender

class Autorec(recommender.PredictRecommender):
    def __init__(self, num_users, num_items,
                 hidden_neuron=500, lambda_value=1,
                 train_epoch=1000, batch_size=1000, optimizer_method='RMSProp',
                 grad_clip=False, base_lr=1e-3, lr_decay=0.99,
                 dropout=0.05, random_seed=0):
        """Create new Autorec recommender."""
        super().__init__()
        self.model = autorec.AutoRec(num_users, num_items,
                             seen_users=set(), seen_items=set(),
                             hidden_neuron=hidden_neuron,
                             dropout=dropout, random_seed=random_seed)
        self.lambda_value = lambda_value
        self.num_users = num_users
        self.num_items = num_items
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.num_batch = int(math.ceil(self.num_users / float(self.batch_size)))
        self.base_lr = base_lr
        self.optimizer_method = optimizer_method
        self.random_seed = random_seed

        self.lr_decay = lr_decay
        self.grad_clip = grad_clip
        np.random.seed(self.random_seed)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def train_model(self, data):
        """
        Trains for all epochs in train_epoch
        """
        self.model.train()
        if self.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)

        elif self.optimizer_method == "RMSProp":
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.base_lr)
        else:
            raise ValueError("Optimizer Key ERROR")

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)

        self.model.to(self.device)
        for epoch in range(self.train_epoch):
            self.train(data, optimizer, scheduler)

    def train(self, data, optimizer, scheduler):
        """Trains for a single epoch"""
        random_perm_doc_idx = np.random.permutation(self.num_items)
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size:]
            elif i < self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size : (i+1) * self.batch_size]

            batch = data[batch_set_idx, :].to(self.device)
            output = self.model.forward(batch)
            mask = self.mask_R[batch_set_idx, :].to(self.device)
            loss = self.model._loss(output, batch,
                             mask, lambda_value=self.lambda_value)

            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            optimizer.step()
            scheduler.step()

    @property
    def name(self):  # noqa: D102
        return 'autorec'

    def _predict(self, user_item):
        return self.model.predict(user_item, self.R.to(self.device))

    def reset(self, users=None, items=None, ratings=None):  # noqa: D102
        self.model.prepare_model()
        super().reset(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        super().update(users, items, ratings)
        for user_item in ratings:
            self.model.seen_users.add(user_item[0])
            self.model.seen_items.add(user_item[1])

        ratings = self._ratings.toarray()
        # item-based autorec expects rows that represent items
        self.R = torch.FloatTensor(ratings.T)
        self.mask_R = torch.FloatTensor(ratings.T).clamp(0, 1)

        self.train_model(self.R)
