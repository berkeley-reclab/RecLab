"""A utility module for loading and manipulating various datasets."""
import collections
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')


def split_ratings(ratings, proportion, shuffle=False):
    """Split a group of ratings into two groups.

    Parameters
    ----------
    ratings : dict
        The ratings to split.
    proportion : float
        The proportion of ratings that will be in the first group. Must be between 0 and 1.
    shuffle : bool
        Whether to shuffle the rating data.

    Returns
    -------
    ratings_1 : OrderedDict
        The first set of ratings.
    ratings_2 : OrderedDict
        The second set of ratings.

    """
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

def read_dataset(name, shuffle=True):
    """Read a dataset as specified by name.

    Returns
    -------
    users : dict
        The dict of all users where the key is the user-id and the value is the user's features.
    items : dict
        The dict of all items where the key is the item-id and the value is the item's features.
    ratings : dict
        The dict of all ratings where the key is a tuple whose first element is the user-id
        and whose second element is the item id. The value is a tuple whose first element is the
        rating value and whose second element is the rating context (in this case an empty array).

    """
    if name == 'ml-100k':
        dir_name = 'ml-100k'
        data_name = 'u.data'
        data_url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        csv_params = dict(sep='\t', header=None, usecols=[0, 1, 2, 3],
                       names=['user_id', 'item_id', 'rating', 'timestamp'])
    elif name == 'ml-10m':
        dir_name = 'ml-10M100K'
        data_name = 'ratings.dat'
        data_url = 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'
        csv_params = dict(sep='::', header=None, usecols=[0, 1, 2, 3],
                       names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
    else:
        raise ValueError('dataset name not recognized')

    data_dir = os.path.join(DATA_DIR, dir_name)
    datafile = os.path.join(data_dir, data_name)
    if not os.path.isfile(datafile):
        os.makedirs(DATA_DIR, exist_ok=True)
        download_location = os.path.join(DATA_DIR, '{}.zip'.format(data_dir))
        urllib.request.urlretrieve(data_url,
                                   filename=download_location)
        with zipfile.ZipFile(download_location, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(download_location)

    data = pd.read_csv(datafile, **csv_params)
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)


    users = {user_id: np.zeros(0) for user_id in np.unique(data['user_id'])}
    items = {item_id: np.zeros(0) for item_id in np.unique(data['item_id'])}

    # Fill the rating array with initial data.
    ratings = {}
    for user_id, item_id, rating in zip(data['user_id'], data['item_id'], data['rating']):
        # TODO: may want to eventually a rating context (e.g. time)
        ratings[user_id, item_id] = (rating, np.zeros(0))

    return users, items, ratings