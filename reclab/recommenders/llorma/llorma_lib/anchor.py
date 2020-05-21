"""Anchor Manager module
"""
import random

import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial import distance_matrix


def _init_anchor_points(data, n_anchor, row_k, col_k):
    """ Helper function that

    Parameters
    ----------
    data : array-like, shape [n_ratings, 3]
        Rating data
        Each row is of the form [user_id, item_id, rating]
    n_anchor : int
        Number of anchor points
    row_k : array-like, shape [n_users, n_users]
        Symmetric kernel matrix where entry (i,j) is
        the similarity between user_i and user_j
    col_k : array-like, shape [n_items, n_items]
        Symmetric kernel matrix where entry (i, j) id
        the similarity between item_i and item_j

    Returns
    -------
    np.ndarray, shape (n_anchor,)
        Array of anchor indices, indexed according
        to their order in the rating data
    """
    user_ids = data[:, 0].astype(np.int64)
    item_ids = data[:, 1].astype(np.int64)

    anchor_idxs = []
    while len(anchor_idxs) < n_anchor:
        anchor_idx = random.randint(0, data.shape[0] - 1)
        if anchor_idx in anchor_idxs:
            continue

        anchor_row = data[anchor_idx]
        uid = int(anchor_row[0])
        iid = int(anchor_row[1])

        k = np.multiply(row_k[uid][user_ids],
                        col_k[iid][item_ids])
        sum_a_of_anchor = np.sum(k)
        if sum_a_of_anchor < 1:
            continue

        #print('>> %10d\t%d' % (anchor_idx, sum_a_of_anchor))
        anchor_idxs.append(anchor_idx)

    return anchor_idxs


def _get_distance_matrix(latent):
    """Helper function to compute a matrix
    of pairwise cosine distances between latent
    factors of a pair of users of a pair of items

    Parameters
    ----------
    latent : array-like, shape (N, latent_dim)
        Matrix of latent factors
        Number of rows is the number of users or items
        Number of columns is the latent dimension

    Returns
    -------
    array-like, shape (N, N)
        Matrix of cosine distances between every
        pair of users (items)
    """
    _normalized_latent = normalize(latent, axis=1)

    d_mat = distance_matrix(_normalized_latent, _normalized_latent)
    assert np.count_nonzero(np.isnan(d_mat)) == 0
    return d_mat


def _get_k_from_distance(d_mat):
    """Helper function to compute kernel matrix from distance matrix

    Parameters
    ----------
    d_mat : array-like, shape [N, N]
        Matrix of cosine distances between every
        pair of users (items)

    Returns
    -------
    np.ndarray, shape [N, N]
        Kernel matrix corresponding to the distance matrix
    """
    m_mat = np.zeros(d_mat.shape)
    m_mat[d_mat < 0.9] = 1
    k_mat = np.multiply(np.subtract(np.ones(d_mat.shape), np.square(d_mat)), m_mat)
    return k_mat

def _get_rbf_k(latent, gamma=None, scaled=True):
    """Helper function to compute scaled
    Gaussian Kernel matrix for latent factors

    Parameters
    ----------
    latent : array-like, shape (N, latent_dim)
        Matrix of latent factors
        Number of rows is the number of users or items
        Number of columns is the latent dimension
    gamma : float, optional
        parameter for the , by default None
    scaled : bool, optional
        if true, the kernel is scaled by the norms of the factors
        by default True
    """

    if gamma is None:
        gamma = 1
    d_mat = _get_distance_matrix(latent)

    rbf_mat = np.exp(-1*gamma*d_mat)
    row_norms = np.linalg.norm(latent, axis=1)
    if scaled:
        norms_mat = np.outer(row_norms, row_norms)
        k_mat = np.multiply(rbf_mat, norms_mat)
    else: k_mat = rbf_mat

    # normalize such that diagonals have value 1
    row_avg = np.mean(k_mat, axis=1, keepdims=True).reshape(-1, 1)
    col_avg = np.mean(k_mat, axis=0, keepdims=True).reshape(1, -1)
    avg = np.mean(k_mat)
    k_mat = k_mat-col_avg-row_avg+2*avg
    k_diag = np.sqrt(np.diagonal(k_mat))
    k_diag_outer = np.outer(k_diag, k_diag)
    k_mat = np.divide(k_mat, k_diag_outer)
    # return (k_mat - 1)*2
    return(k_mat)



def _get_ks_from_latents(row_latent, col_latent):
    """Helper function to get kernels

    Parameters
    ----------
    row_latent : array-like, shape (N_users, rank)
        Matrix of latent factors corresponding to users
    col_latent : array-like, shape (N_items, rank)
        Matrix of latent factors corresponding to items

    Returns
    -------
    (row_k, col_k): array-like, (N_users, N_users), (N_items, N_items)
        Returns two square matrices corresponding to similarity kernels
        row_k: entry (i,j) is the similarity between user_i and user_j
        col_k: entry (i,j) is the similarity between item_i and item_j
    """
    # row_d = _get_distance_matrix(row_latent)
    # col_d = _get_distance_matrix(col_latent)

    # row_k = _get_k_from_distance(row_d)
    # col_k = _get_k_from_distance(col_d)

    row_k = _get_rbf_k(row_latent)
    col_k = _get_rbf_k(col_latent)

    return row_k, col_k


class AnchorManager:
    """ AnchorManager class

    Parameters
    ----------
    n_anchor : int
        number of anchor points
    batch_manager : obj: BatchManager
        an instance of BatchManager class
    row_latent_init : array-like, shape (n_users, latent_dim)
        Matrix of latent factors for users.
        Typically this is set to factors pre-trained in a
        pre-train Matrix Factorization step
    col_latent_init : array-like, shape (n_item, latent_dim)
        Matrix of latent factors for items.
        Typically this is set to factors pre-trained in a
        pre-train Matrix Factorization step
    """

    def __init__(
            self,
            n_anchor,
            batch_manager,
            row_latent_init,
            col_latent_init,
            kernel_fun):
        """ Instantiate an AnchorManager
        """

        train_data = batch_manager.train_data

        row_latent = row_latent_init
        col_latent = col_latent_init

        if kernel_fun is None:
            row_k, col_k = _get_ks_from_latents(row_latent, col_latent)
        else:
            row_k = kernel_fun(row_latent)
            col_k = kernel_fun(col_latent)

        anchor_idxs = _init_anchor_points(train_data, n_anchor, row_k, col_k)
        assert len(anchor_idxs) == n_anchor
        anchor_points = train_data[anchor_idxs]

        self.train_data = train_data
        self.valid_data = batch_manager.valid_data
        self.test_data = batch_manager.test_data

        self.anchor_idxs = anchor_idxs
        self.anchor_points = anchor_points

        self.row_k = row_k
        self.col_k = col_k

    def get_k(self, anchor_idx, user_item_data):
        """Returns the Kernel similarity between the
        anchor user_item pair and the user_item pairs
        in the user_item data

        Parameters
        ----------
        anchor_idx : Array-like, shape (2,)
            (user_id, item_id) of the anchor point
        user_item_data : Array-like, shape (N_ratings, >2)
            Array where first 2 columns are (user_id, item_id) pairs

        Returns
        -------
        np.ndarray, shape (N_ratings,)
            Returns an array of kernel weights corresponding to
            the chosen anchor for each user_item pair in the data
        """
        row_k = self.row_k
        col_k = self.col_k
        anchor_point = self.anchor_points[anchor_idx]

        anchor_uid = int(anchor_point[0])
        anchor_iid = int(anchor_point[1])

        user_ids = user_item_data[:, 0].astype(np.int64)
        item_ids = user_item_data[:, 1].astype(np.int64)

        return np.multiply(row_k[anchor_uid][user_ids], col_k[anchor_iid][item_ids])

    def get_train_k(self, anchor_idx):
        """ Get Kernel matrix of the train_data of a given anchor

        Parameters
        ----------
        anchor_idx : Array-like, shape (2,)
            (user_id, item_id) of the anchor point

        Returns
        -------
        np.ndarray, shape (N_ratings,)
            Returns an array of kernel weights corresponding to
            the chosen anchor for each user_item pair in the train data
        """
        return self.get_k(anchor_idx, self.train_data)

    def get_valid_k(self, anchor_idx):
        """ Get Kernel matrix of the validation_data of a given anchor

        Parameters
        ----------
        anchor_idx : Array-like, shape (2,)
            (user_id, item_id) of the anchor point

        Returns
        -------
        np.ndarray, shape (N_ratings,)
            Returns an array of kernel weights corresponding to
            the chosen anchor for each user_item pair in the valid data
        """
        return self.get_k(anchor_idx, self.valid_data)

    def get_test_k(self, anchor_idx):
        """ Get Kernel matrix of the test_data of a given anchor

        Parameters
        ----------
        anchor_idx : Array-like, shape (2,)
            (user_id, item_id) of the anchor point

        Returns
        -------
        np.ndarray, shape (N_ratings,)
            Returns an array of kernel weights corresponding to
            the chosen anchor for each user_item pair in the test data
        """
        return self.get_k(anchor_idx, self.test_data)
