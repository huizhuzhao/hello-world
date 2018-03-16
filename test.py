#!/usr/bin/env python
# encoding: utf-8
# Created Time: 2017年01月21日 星期六 19时30分16秒

import logging
import sys
sys.path.append('/home/xtalpi/github/cnn_graph')
from lib import graph
import numpy as np
import sklearn.metrics
import sklearn.neighbors
import scipy.spatial.distance
from foundation.utils import numpy_utils, generic_utils


def grid(m, dtype=np.float32):
    """
    return
        z: 2D array with shape=[M, 2]
        m pointers in [0, 1], reture meshgrid of xx, yy
    """
    M = m**2
    x = np.linspace(0, 1, m, dtype=dtype)
    y = np.linspace(0, 1, m, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), dtype)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)

    return z


def distance_scipy_spatial(z, k=4, metric='euclidean'):
    d = scipy.spatial.distance.pdist(z, metric)
    d = scipy.spatial.distance.squareform(d)

    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]

    return d, idx


def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)
    # nearest k neighbors indices
    idx = np.argsort(d)[:, 1:k+1]
    # nearest k neighbors values
    d.sort()
    d = d[:, 1:k+1]

    return d, idx

def distance_lshforest(z, k=4, metric='cosine'):
    assert metric is 'cosine'
    lshf = sklearn.neighbors.LSHForest()
    lshf.fit(z)
    dist, idx = lshf.kneighbors(z, n_neighbors=k+1)
    print(dist.shape, idx.shape)


def adjacency(dist, idx):
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

    sigma2 = np.mean(dist[:, -1]) ** 2 # maximu distance
    dist = np.exp(- dist ** 2 / sigma2)
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M * k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))
    
    W.setdiag(0)

    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger) # now W is a symmetric matrix
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix

    return W

def laplacian(W, normalized=True):
    d = W.sum(axis=0)
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        delta = np.spacing(np.array(0, W.dtype))
        d = d + delta
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def grid_graph(m, corners=False):
    """
    返回 graph 的连接矩阵(symmetric, nearest k neighbors)
    """
    num_edges = 8
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=8, metric='euclidean')
    A = graph.adjacency(dist, idx)
    print('A.nnz: {}'.format(A.nnz))
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz // 2, num_edges * (m**2) // 2))
    return A

def replace_random_edges(A, noise_level):
    M, M = A.shape
    n = int(noise_level * A.nnz // 2)
    indices = np.random.permutation(A.nnz // 2)[:n]

    rows = np.random.randint(0, M, n)
    cols = np.random.randint(0, M, n)
    vals = np.random.uniform(0, 1, n)
    assert len(indices) == len(rows) == len(cols) == len(vals)

    A_coo = scipy.sparse.triu(A, format='coo')
    assert A_coo.nnz == A.nnz // 2, ("{} {} do not match".format(A_coo.nnz, A.nnz // 2))
    assert A_coo.nnz >= n
    A = A.tolil() # to linked list format
    for idx, row, col, val in zip(indices, rows, cols, vals):
        old_row = A_coo.row[idx]
        old_col = A_coo.col[idx]
        A[old_row, old_col] = 0
        A[old_col, old_row] = 0
        A[row, col] = 1
        A[col, row] = 1

    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def main():
    logger = generic_utils.get_logger(level='debug')
    A = grid_graph(28, corners=False)
    A = replace_random_edges(A, 0.3)


if __name__ == '__main__':
	main()
