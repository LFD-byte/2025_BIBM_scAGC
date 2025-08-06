import os
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.random import seed
from preprocess import *
from utils import *
# python train.py --dataname Adam --highly_genes 500 --pretrain_epochs 1000 --maxiter 300
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn import metrics
import scipy.io as scio
import hydra
from omegaconf import DictConfig, OmegaConf
from tensorflow.keras import backend as K # 导入后端
seed(1)
tf.random.set_seed(1)

from scipy import sparse as sp


# Remove warnings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from scagc import SCAGC
from loss import *
from graph_function import *

# Compute cluster centroids, which is the mean of all points in one cluster.
def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def log_result(cfg, scores, filename="result.csv"):
    result = {
        **scores,
        'dataset': cfg.dataset.dataname,
    }
    df = pd.DataFrame([result])
    print("Logging results to:", filename)
    df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

def collect_results(root_dir="multirun/2025-07-20"):
    all_records = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == "result.csv":
                df = pd.read_csv(os.path.join(subdir, file))
                all_records.append(df)
    combined = pd.concat(all_records, ignore_index=True)
    return combined

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("TensorFlow version:", tf.__version__)
    print("Eager Execution is enabled:", tf.executing_eagerly())
    K.clear_session()
    print("current configuration:\n", OmegaConf.to_yaml(cfg))

    # ["Adam","Bach","Klein","Muraro","Plasschaert","Pollen","Quake_10x_Bladder","Quake_10x_Limb_Muscle",
    # "Quake_10x_Spleen","Quake_10x_Trachea","Quake_Smart-seq2_Diaphragm","Quake_Smart-seq2_Heart",
    # "Quake_Smart-seq2_Limb_Muscle","Quake_Smart-seq2_Lung","Quake_Smart-seq2_Trachea","Romanov",
    # "Wang_Lung","Young"]

    # Load data
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training.gpu_option
    x, y = prepro('dataset/' + cfg.dataset.dataname + '/data.h5')
                
    x = np.ceil(x).astype(np.int32)
    cluster_number = int(max(y) - min(y) + 1)
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    adata = normalize(adata, copy=True, highly_genes=cfg.dataset.highly_variable_genes, size_factors=True, normalize_input=True, logtrans_input=True)
    count = adata.X
    
    # Build model
    adj, adj_n = get_adj(count, k=cfg.training.k)
    model = SCAGC(count, adj=adj, adj_n=adj_n)

    # Pre-training
    model.pre_train(epochs=cfg.training.pretrain_epochs, contra_temperature=cfg.training.temperature, k=cfg.training.k,
                    gumbel_temperature=cfg.training.gumbel_temperature)

    Y = model.embedding(count, model.adj_n)

    from sklearn.cluster import SpectralClustering
    labels = SpectralClustering(n_clusters=cluster_number,affinity="precomputed", assign_labels="discretize",random_state=0).fit_predict(adj)
    centers = computeCentroids(Y, labels)
    
    scores = {}
    # Clustering training
    Cluster_predicted = model.alt_train(y, epochs=cfg.training.maxiter, centers=centers, 
                                                    contra_temperature=cfg.training.temperature, k=cfg.training.k,
                                                    gumbel_temperature=cfg.training.gumbel_temperature)
    if y is not None:
        acc = np.round(cluster_acc(y, Cluster_predicted.y_pred), 5)
        y = list(map(int, y))
        Cluster_predicted.y_pred = np.array(Cluster_predicted.y_pred)
        nmi = np.round(metrics.normalized_mutual_info_score(y, Cluster_predicted.y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, Cluster_predicted.y_pred), 5)
        print('ACC= %.4f, NMI= %.4f, ARI= %.4f'
            % (acc, nmi, ari))
        scores['acc'] = acc
        scores['nmi'] = nmi
        scores['ari'] = ari

    fname = f"result-scagc-data_{cfg.dataset.dataname}-temperature_{cfg.training.temperature}-epochs1_{cfg.training.pretrain_epochs}-epochs2_{cfg.training.maxiter}.csv"
    
    print(fname)
    log_result(cfg, scores, filename=fname)
    print("Results logged to CSV file.")

if __name__ == "__main__":
    scores = main()
