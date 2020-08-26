"""Pytorch implementation of AutoRec recommender."""

import math
import numpy as np
import torch

from .autorec_lib import autorec
from .. import recommender


class Autorec(recommender.PredictRecommender):
    """The Autorec recommender.

    Parameters
    ----------
    num_users : int
        Number of users in the environment.
    num_items : int
        Number of items in the environment.
    hidden_neuron : int
        Output dimension of hidden layer.
    lambda_value : float
        Coefficient for regularization while training layers.
    train_epoch : int
        Number of epochs to train for each call.
    batch_size : int
        Batch size during initial training phase.
    optimizer_method : str
        Optimizer for training model; either Adam or RMSProp.
    grad_clip : bool
        Set to true to clip gradients to [-5, 5].
    base_lr : float
        Base learning rate for optimizer.
    lr_decay : float
        Rate for decaying learning rate during training.
    dropout : float
        Probability to initialize dropout layer. Set to 0 for no dropout.
    random_seed : int
        Random seed to reproduce results.

    """

    def __init__(self, num_users, num_items,
                 hidden_neuron=500, lambda_value=1,
                 train_epoch=1000, batch_size=1000, optimizer_method='RMSProp',
                 grad_clip=False, base_lr=1e-3, lr_decay=1e-2,
                 dropout=0.05, random_seed=0):
        """Create new Autorec recommender."""
        super().__init__()

        # We only want the function arguments so remove class related objects.
        self._hyperparameters.update(locals())
        del self._hyperparameters['self']
        del self._hyperparameters['__class__']

        self.model = autorec.AutoRec(num_users,
                                     num_items,
                                     seen_users=set(),
                                     seen_items=set(),
                                     hidden_neuron=hidden_neuron,
                                     dropout=dropout,
                                     random_seed=random_seed)
        self.lambda_value = lambda_value
        self.num_users = num_users
        self.num_items = num_items
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.num_batch = int(math.ceil(self.num_items / float(self.batch_size)))
        self.base_lr = base_lr
        self.optimizer_method = optimizer_method
        self.random_seed = random_seed

        self.lr_decay = lr_decay
        self.grad_clip = grad_clip
        np.random.seed(self.random_seed)
        # pylint: disable=no-member
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def train_model(self, data):
        """Train for all epochs in train_epoch."""
        self.model.train()
        if self.optimizer_method == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)

        elif self.optimizer_method == 'RMSProp':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.base_lr)
        else:
            raise ValueError('Optimizer Key ERROR')

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=self.lr_decay)

        self.model.to(self.device)
        for epoch in range(self.train_epoch):
            self.train(data, optimizer, scheduler)

    def train(self, data, optimizer, scheduler):
        """Train for a single epoch."""
        random_perm_doc_idx = np.random.permutation(self.num_items)
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size:]
            elif i < self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size:(i+1) * self.batch_size]

            batch = data[batch_set_idx, :].to(self.device)
            output = self.model.forward(batch)
            mask = self.mask_ratings[batch_set_idx, :].to(self.device)
            loss = self.model.loss(output,
                                   batch,
                                   mask,
                                   lambda_value=self.lambda_value)

            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            optimizer.step()
            scheduler.step()

    @property
    def name(self):  # noqa: D102
        return 'autorec'

    def _predict(self, user_item):
        self.model = self.model.eval()
        return self.model.predict(user_item, self.ratings.to(self.device))

    def reset(self, users=None, items=None, ratings=None):  # noqa: D102
        self.model.prepare_model()
        super().reset(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        super().update(users, items, ratings)
        self.model.prepare_model()
        self.model = self.model.train()
        for user_item in ratings:
            self.model.seen_users.add(user_item[0])
            self.model.seen_items.add(user_item[1])

        ratings = self._ratings.toarray()
        # Item-based autorec expects rows that represent items
        # pylint: disable=no-member
        self.ratings = torch.FloatTensor(ratings.T)
        # pylint: disable=no-member
        self.mask_ratings = torch.FloatTensor(ratings.T).clamp(0, 1)

        self.train_model(self.ratings)
