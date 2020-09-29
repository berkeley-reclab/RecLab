"""A utility module for loading and manipulating various datasets."""
import collections
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse


DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')


def read_dataset(name, shuffle=True):
    """Read a dataset as specified by name.

    Parameters
    ----------
    name : str
        The name of the dataset. Must be one of: 'ml-100k', 'ml-10m', 'citeulike-a',
        'pinterest', or 'lastfm'.
    shuffle : bool, optional
        A flag to indicate whether the dataset should be shuffled after loading,
        true by default.

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
    data = get_data(name)

    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)

    users = {user_id: np.zeros(0) for user_id in np.unique(data['user_id'])}
    items = {item_id: np.zeros(0) for item_id in np.unique(data['item_id'])}

    # Fill the rating array with initial data.
    ratings = {}
    for user_id, item_id, rating in zip(data['user_id'], data['item_id'], data['rating']):
        # TODO: may want to eventually a rating context depending on dataset (e.g. time)
        ratings[user_id, item_id] = (rating, np.zeros(0))

    return users, items, ratings


def read_bandit_dataset(name):
    """Read a bandit dataset as specified by name.

    Parameters
    ----------
    name : str
        The name of the dataset. Must be one of: 'wiki10-31k'.

    Returns
    -------
    features : scipy.sparse.dok_matrix
        The features at each timestep.
    ratings : scipy.sparse.dok_matrix
        The ratings at each timestep.

    """
    if name == 'wiki10-31k':
        with open_zipped(zipped_dir_name='wiki10-31k',
                         data_name='features.npz',
                         data_url='https://kkrauth.s3-us-west-2.amazonaws.com/wiki10-31k.zip',
                         mode='rb') as feature_file:
            features = scipy.sparse.load_npz(feature_file).tocsr()

        with open_zipped(zipped_dir_name='wiki10-31k',
                         data_name='ratings.npz',
                         data_url='https://kkrauth.s3-us-west-2.amazonaws.com/wiki10-31k.zip',
                         mode='rb') as ratings_file:
            ratings = scipy.sparse.load_npz(ratings_file).tocsr()
    else:
        raise ValueError('Dataset name not recognized.')

    return features, ratings


def split_ratings(ratings, proportion, shuffle=False, seed=None):
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
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(iterator)

    for i, (key, val) in enumerate(iterator):
        if i < split_1_end:
            split_1[key] = val
        else:
            split_2[key] = val

    return split_1, split_2


def read_zipped_csv(zipped_dir_name, data_name, data_url, csv_params):
    """Locate or download zipped file and load csv into DataFrame.

    Parameters
    ----------
    zipped_dir_name : str
        The directory within the downloaded zip.
    data_name : str
        The name of the data file to be loaded from the directory.
    data_url : str
        The location of the download.
    csv_params : str
        Parameters for loading csv into DataFrame.

    Returns
    -------
    data : DataFrame
        Dataset of interest.

    """
    data_file = os.path.join(DATA_DIR, zipped_dir_name, data_name)
    fetch_zip(zipped_dir_name, data_url)
    return pd.read_csv(data_file, **csv_params)


def open_zipped(zipped_dir_name, data_name, data_url, mode):
    """Download a zipped file and open it.

    Parameters
    ----------
    zipped_dir_name : str
        The directory within the downloaded zip.
    data_name : str
        The name of the data file to be loaded from the directory.
    data_url : str
        The location of the download.
    mode: str
        The mode to open the file in.

    Returns
    -------
    file : file
        The file of interest.

    """
    data_file = os.path.join(DATA_DIR, zipped_dir_name, data_name)
    fetch_zip(zipped_dir_name, data_url)
    return open(data_file, mode)


def fetch_zip(zipped_dir_name, data_url):
    """Download a zipped directory and extract it.

    Parameters
    ----------
    zipped_dir_name : str
        The directory within the downloaded zip.
    data_url : str
        The location of the download.

    """
    data_dir = os.path.join(DATA_DIR, zipped_dir_name)
    if not os.path.isdir(data_dir):
        os.makedirs(DATA_DIR, exist_ok=True)

        download_location = os.path.join('{}.zip'.format(data_dir))
        urllib.request.urlretrieve(data_url,
                                   filename=download_location)
        with zipfile.ZipFile(download_location, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(download_location)


def find_npz(dir_name, data_name, data_url, np_params):
    """Locate or download npz file and load into DataFrame.

    Parameters
    ----------
    dir_name : str
        The directory to put the .npz file.
    data_name : str
        The name of the .npz file.
    data_url : str
        The location of the download.
    csv_params : str
        Parameters for loading the numpy array into DataFrame.

    Returns
    -------
    data : DataFrame
        Dataset of interest.

    """
    download_dir = os.path.join(DATA_DIR, dir_name)
    datafile = os.path.join(download_dir, data_name)
    if not os.path.isfile(datafile):
        os.makedirs(download_dir, exist_ok=True)
        urllib.request.urlretrieve(data_url, filename=datafile)
    data_np = np.load(datafile, allow_pickle=True)['train_data']
    data = pd.DataFrame(data_np, **np_params)
    # TODO: deal better with implicit ratings
    data['rating'] = 1
    return data


def get_data(name):
    """Read a dataset specified by name into pandas dataframe.

    Parameters
    ----------
    name : str
        The name of the dataset. Must be one of: 'ml-100k', 'ml-10m', 'ml-1m',
        'citeulike-a', 'pinterest', or 'lastfm'.

    Returns
    -------
    data : DataFrame
        Dataset of interest.

    """
    if name == 'ml-100k':
        zipped_dir_name = 'ml-100k'
        data_name = 'u.data'
        data_url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        csv_params = dict(sep='\t', header=None, usecols=[0, 1, 2, 3],
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
        data = read_zipped_csv(zipped_dir_name, data_name, data_url, csv_params)
    elif name == 'ml-10m':
        zipped_dir_name = 'ml-10M100K'
        data_name = 'ratings.dat'
        data_url = 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'
        csv_params = dict(sep='::', header=None, usecols=[0, 1, 2, 3],
                          names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
        data = read_zipped_csv(zipped_dir_name, data_name, data_url, csv_params)
    elif name == 'ml-1m':
        zipped_dir_name = 'ml-1m'
        data_name = 'ratings.dat'
        data_url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        csv_params = dict(sep='::', header=None, usecols=[0, 1, 2, 3],
                          names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
        data = read_zipped_csv(zipped_dir_name, data_name, data_url, csv_params)
    elif name == 'citeulike-a':
        dir_name = 'citeulike-a'
        data_name = 'data.npz'
        data_url = ('https://raw.githubusercontent.com/tebesu/CollaborativeMemoryNetwork/'
                    'master/data/citeulike-a.npz')
        np_params = dict(columns=['user_id', 'item_id'])
        data = find_npz(dir_name, data_name, data_url, np_params)
    elif name == 'pinterest':
        dir_name = 'pinterest'
        data_name = 'data.npz'
        data_url = ('https://raw.githubusercontent.com/tebesu/CollaborativeMemoryNetwork/'
                    'master/data/pinterest.npz')
        np_params = dict(columns=['user_id', 'item_id'])
        data = find_npz(dir_name, data_name, data_url, np_params)
    elif name == 'lastfm':
        data_name = 'lastfm-dataset-1K/lfm1k-play-counts.csv'
        csv_params = dict(header=0, usecols=[0, 1, 2],
                          names=['user_id', 'item_id', 'rating'])
        datafile = os.path.join(DATA_DIR, data_name)
        try:
            data = pd.read_csv(datafile, **csv_params)
            # log transform for better scaling
            data['rating'] = np.log(1 + data['rating'])
            # TODO: remove artists with less than 50 total listens?
            # otherwise should probably retrain for hyperparameter tuning...
        except FileNotFoundError as error:
            print(('LastFM data must be downloaded and preprocessed locally, '
                   'get files from https://drive.google.com/open?id=1qxmsQHe'
                   'D8O-81CbHxvaFP8omMvMxgEh0'))
            raise error
    else:
        raise ValueError('dataset name not recognized')
    return data


def get_time_split_dataset(name, shuffle=True, binarize=False):
    """Get a time-based test/train split of a dataset as specified by name.

    Parameters
    ----------
    name : str
        The name of the dataset. Must be one of: 'ml-100k', 'ml-10m', 'citeulike-a',
        'pinterest', or 'lastfm'.
    shuffle : bool, optional
        A flag to indicate whether the dataset should be shuffled after loading,
        true by default.
    binarize : bool, optional
        A flag to indicate whether to binarize the ratings to be 0 or 1,
        true by default.

    Returns
    -------
    users : dict
        The dict of all users where the key is the user-id and the value is the user's features.
    items : dict
        The dict of all items where the key is the item-id and the value is the item's features.
    train_ratings : dict
        The dict of all training ratings.
    test_ratings : dict
        The dict of all testing ratings.

    """
    data = get_data(name)
    if binarize:
        data['rating'] = 1

    users = {user_id: np.zeros(0) for user_id in np.unique(data['user_id'])}
    items = {item_id: np.zeros(0) for item_id in np.unique(data['item_id'])}

    # Add final rating to test set
    test_idx = []
    for uid in np.unique(data['user_id']):
        last_rating_idx = data[data['user_id'] == uid]['timestamp'].idxmax()
        test_idx.append(last_rating_idx)
    data_test = data.loc[test_idx]
    data_train = data.drop(test_idx)

    # Shuffle remaining data
    if shuffle:
        data_train = data_train.sample(frac=1).reset_index(drop=True)

    # Fill the rating array with initial data.
    train_ratings = {}
    for user_id, item_id, rating in zip(data_train['user_id'], data_train['item_id'],
                                        data_train['rating']):
        # TODO: may want to eventually a rating context depending on dataset (e.g. time)
        train_ratings[user_id, item_id] = (rating, np.zeros(0))

    # Fill the rating array with initial data.
    test_ratings = {}
    for user_id, item_id, rating in zip(data_test['user_id'], data_test['item_id'],
                                        data_test['rating']):
        # TODO: may want to eventually a rating context depending on dataset (e.g. time)
        test_ratings[user_id, item_id] = (rating, np.zeros(0))

    return users, items, train_ratings, test_ratings
