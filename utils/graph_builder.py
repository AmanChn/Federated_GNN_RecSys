import numpy as np
import scipy.sparse as sp
import torch


def build_adj_matrix(train_df, n_users, n_items):
    """
    Build user-item bipartite adjacency matrix
    """

    users = train_df['user'].values
    items = train_df['item'].values

    # shift item indices
    items = items + n_users

    # edges
    rows = np.concatenate([users, items])
    cols = np.concatenate([items, users])

    data = np.ones(len(rows))

    adj_matrix = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_users + n_items, n_users + n_items)
    )

    return adj_matrix


def normalize_adj_matrix(adj):
    """
    Compute normalized adjacency matrix:
    D^(-1/2) A D^(-1/2)
    """

    rowsum = np.array(adj.sum(axis=1)).flatten()

    d_inv = np.power(rowsum, -0.5)
    d_inv[np.isinf(d_inv)] = 0.0

    d_mat = sp.diags(d_inv)

    norm_adj = d_mat.dot(adj).dot(d_mat)

    return norm_adj


def convert_to_torch_sparse(adj):
    """
    Convert scipy sparse matrix to torch sparse tensor
    """

    adj = adj.tocoo()

    indices = torch.from_numpy(
        np.vstack((adj.row, adj.col)).astype(np.int64)
    )

    values = torch.from_numpy(adj.data.astype(np.float32))

    shape = torch.Size(adj.shape)

    return torch.sparse.FloatTensor(indices, values, shape)