import collections
import os
import shutil

import numpy as np
import pandas as pd
import urllib.request
import zipfile


DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


def split_ratings(ratings, proportion, shuffle=False):
    split_1 = collections.OrderedDict()
    split_2 = collections.OrderedDict()
    split_1_end = int(proportion * len(ratings))
    iterator = list(ratings.items())

    if shuffle:
        np.random.shuffle(iterator)

    for i, (key, val) in enumerate(iterator):
        if i < split_1_end:
            split_1[key] = val
        else:
            split_2[key] = val

    return split_1, split_2


def read_movielens100k():
    movielens_dir = os.path.join(DATA_DIR, 'ml-100k')
    datafile = os.path.join(movielens_dir, 'u.data')
    if not os.path.isfile(datafile):
        os.makedirs(DATA_DIR, exist_ok=True)
        download_location = os.path.join(DATA_DIR, 'ml-100k.zip')
        urllib.request.urlretrieve('http://files.grouplens.org/datasets/movielens/ml-100k.zip',
                                   filename=download_location)
        with zipfile.ZipFile(download_location, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(download_location)

    data = pd.read_csv(datafile, sep='\t', header=None, usecols=[0, 1, 2, 3],
                       names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Shifting user and movie indexing.
    data['user_id'] -= 1
    data['item_id'] -= 1

    # Validating data assumptions.
    assert len(data) == 100000

    users = {user_id: np.zeros(0) for user_id in np.unique(data['user_id'])}
    items = {item_id: np.zeros(0) for item_id in np.unique(data['item_id'])}

    # Fill the rating array with initial data.
    ratings = {}
    for user_id, item_id, rating in zip(data['user_id'], data['item_id'], data['rating']):
        # TODO: may want to eventually add time as a rating context
        ratings[user_id, item_id] = (rating, np.zeros(0))

    return users, items, ratings
