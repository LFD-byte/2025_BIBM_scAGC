# scAGC: Learning Adaptive Cell Graphs with Contrastive Guidance for Single-Cell Clustering

Accurate cell type annotation is a crucial step in analyzing single-cell RNA sequencing (scRNA-seq) data, which provides valuable insights into cellular heterogeneity. However, due to the high dimensionality and prevalence of zero elements in scRNA-seq data, traditional clustering methods face significant statistical and computational challenges. While some advanced methods use graph neural networks to model cell-cell relationships, they often depend on static graph structures that are sensitive to noise and fail to capture the long-tailed distribution inherent in single-cell populations.
To address these limitations, we propose scAGC, a single-cell clustering method that learns adaptive cell graphs with contrastive guidance. Our approach optimizes feature representations and cell graphs simultaneously in an end-to-end manner. Specifically, we introduce a topology-adaptive graph autoencoder that leverages a differentiable Gumbel-Softmax sampling strategy to dynamically refine the graph structure during training. This adaptive mechanism mitigates the problem of a long-tailed degree distribution by promoting a more balanced neighborhood structure. To model the discrete, over-dispersed, and zero-inflated nature of scRNA-seq data, we integrate a Zero-Inflated Negative Binomial (ZINB) loss for robust feature reconstruction. Furthermore, a contrastive learning objective is incorporated to regularize the graph learning process and prevent abrupt changes in the graph topology, ensuring stability and enhancing convergence. Comprehensive experiments on 9 real scRNA-seq datasets demonstrate that scAGC consistently outperforms other state-of-the-art methods, yielding the best NMI and ARI scores on 9 and 7 datasets, respectively.

## Dependencies

```bash
conda create -n scagc python=3.7.12

conda activate scagc

conda install cudatoolkit=11.2 cudnn -c conda-forge

pip install -r requirements.txt
```

## Dataset download

The datasets can be download from [Google drive](https://drive.google.com/drive/folders/1cZUi7ZCYZOXh3HScashjpYDQXZ-ghrJh?usp=sharing).

## Model training

```bash
chmod 777 run.sh
./run.sh
```

## Evaluations

To compare different clustering methods, we use NMI and ARI.
