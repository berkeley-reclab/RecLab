"""Tests for the TopPop recommender."""
import collections

import numpy as np

from reclab.recommenders import TopPop


def test_top_pop_one_step():
    """Test a single recommendation step."""
    users = {0: np.array(0),
             1: np.array(0),
             2: np.array(0)}
    items = {0: np.array(0),
             1: np.array(0),
             2: np.array(0)}
    ratings = {(0, 0): (5, np.array(0)),
               (0, 1): (4, np.array(0)),
               (1, 1): (4, np.array(0)),
               (1, 2): (3, np.array(0))}
    user_contexts = collections.OrderedDict([(0, np.array(0)),
                                             (1, np.array(0)),
                                             (2, np.array(0))])

    model = TopPop()
    model.reset(users, items, ratings)
    recs, _ = model.recommend(user_contexts, 1)
    assert recs.shape == (3, 1)
    assert recs[0, 0] == 2
    assert recs[1, 0] == 0
    assert recs[2, 0] == 0


def test_top_pop_multi_step():
    """Test multiple rounds of recommending and rating."""
    users = {0: np.array(0),
             1: np.array(0)}
    items = {0: np.array(0),
             1: np.array(0),
             2: np.array(0)}
    ratings = {(0, 0): (5, np.array(0)),
               (1, 1): (3, np.array(0))}
    user_contexts = collections.OrderedDict([(0, np.array(0)),
                                             (1, np.array(0))])

    model = TopPop()
    model.reset(users, items, ratings)
    recs, _ = model.recommend(user_contexts, 1)
    assert recs.shape == (2, 1)
    assert recs[0, 0] == 1
    assert recs[1, 0] == 0
    user_contexts[2] = np.array(0)
    model.update(users={2: np.array(0)},
                 ratings={(0, 1): (5, np.array(0)),
                          (1, 0): (1, np.array(0))})
    recs, _ = model.recommend(user_contexts, 1)
    assert recs.shape == (3, 1)
    assert recs[0, 0] == 2
    assert recs[1, 0] == 2
    assert recs[2, 0] == 1
