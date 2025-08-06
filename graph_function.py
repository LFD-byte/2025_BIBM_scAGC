import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph
from utils import dopca
import scanpy as sc
from anndata import AnnData

def get_adj(count, k=15, pca=50, mode="connectivity"):
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()

    adj_sym_or = adj + adj.T

    # A + A.T 可能会使某些位置的值变成2 (如果A和B互为邻居)
    # 我们需要将其变回二进制矩阵，表示“连接”或“未连接”
    adj_sym_or[adj_sym_or > 1] = 1

    adj = adj_sym_or

    adj_n = norm_adj(adj)
    return adj, adj_n


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


