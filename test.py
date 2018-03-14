#!/usr/bin/env python
# encoding: utf-8
# Created Time: 2017年01月21日 星期六 19时30分16秒

import numpy as np
import sklearn.metrics
import sklearn.neighbors
import scipy.spatial.distance
from foundation.utils import numpy_utils


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


def main():
    z = grid(10)
    d1, idx1 = distance_sklearn_metrics(z)
    d2, idx2 = distance_scipy_spatial(z)
    W = adjacency(d1, idx1)
    laplacian(W, True) 



if __name__ == '__main__':
	main()
