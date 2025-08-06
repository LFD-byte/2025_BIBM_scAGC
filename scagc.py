import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MSE, KLD
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda
from spektral.layers import TAGConv, GATConv, GCNConv, GraphSageConv
from tensorflow.keras.initializers import GlorotUniform
from layers import *
import tensorflow_probability as tfp
import numpy as np
from sklearn import metrics
from loss import poisson_loss, NB, ZINB, dist_loss
from spektral.transforms.gcn_filter import GCNFilter
from spektral.data import Graph

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


MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

def gumbel_topk_sampling_tf(logits, k, temperature=1.0, hard=True):
    gumbels = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), minval=0, maxval=1) + 1e-20) + 1e-20)
    gumbel_logits = (logits + gumbels) / temperature
    y_soft = tf.nn.softmax(gumbel_logits, axis=-1)
    if hard:
        _, indices = tf.math.top_k(y_soft, k, sorted=False)
        y_hard = tf.zeros_like(logits)
        batch_range = tf.range(tf.shape(logits)[0])
        batch_idx = tf.repeat(batch_range, k)
        indices_flat = tf.reshape(indices, [-1])
        full_indices = tf.stack([batch_idx, indices_flat], axis=1)
        y_hard = tf.tensor_scatter_nd_update(y_hard, full_indices, tf.ones(tf.shape(indices_flat)))
        A_soft = tf.stop_gradient(y_hard - y_soft) + y_soft

        A_soft_sym_or = A_soft + tf.transpose(A_soft)
        A_soft = tf.minimum(A_soft_sym_or, 1.0)

    else:
        A_soft = y_soft
    return A_soft

def degree_power_tf(A, k):
    """
    计算度矩阵的k次幂。
    
    参数:
        A (tf.Tensor): 输入的邻接矩阵。
        k (float): 幂指数。

    返回:
        tf.Tensor: 对角度矩阵 D^k。
    """
    # 1. 计算每个节点的度（沿行求和）
    degrees = tf.reduce_sum(A, axis=1)

    # 2. 计算度的 k 次幂
    # 注意：当度为0且k为负数时，tf.pow(0, k)会得到inf（无穷大）
    powered_degrees = tf.pow(degrees, k)
    
    # 3. 将无穷大的值替换为0，避免计算错误
    # 这对应原始代码中的 degrees[np.isinf(degrees)] = 0.
    safe_powered_degrees = tf.where(tf.math.is_inf(powered_degrees), 0.0, powered_degrees)

    # 4. 创建对角矩阵 D^k
    D_k = tf.linalg.diag(safe_powered_degrees)
    
    return D_k

def norm_adj_tf(A):
    """
    对邻接矩阵 A 进行对称归一化 D^(-1/2) * A * D^(-1/2)。

    参数:
        A (tf.Tensor): 输入的邻接矩阵。

    返回:
        tf.Tensor: 归一化后的邻接矩阵。
    """
    # 1. 确保输入是 float32 类型的张量，以便进行数学运算
    A = tf.cast(A, dtype=tf.float32)

    # 2. 计算 D^(-1/2)
    normalized_D = degree_power_tf(A, -0.5)

    # 3. 执行对称归一化: D^(-1/2) * A * D^(-1/2)
    # 在 TensorFlow 中，矩阵乘法使用 tf.matmul() 或 @ 运算符
    output = normalized_D @ A @ normalized_D
    
    return output

# 1. 创建 EarlyStopper 类
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        """
        Args:
            patience (int): 在验证损失没有改善后，等待多少个 epoch。
            min_delta (float): 被认为是“改善”的最小变化量。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_performance = 0

    def early_stop(self, current_performance):
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.counter = 0
        elif current_performance < (self.best_performance + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

import tensorflow as tf

def info_nce_loss(z1, z2, temperature=0.1):
    """
    计算InfoNCE自监督对比损失。

    Args:
        z1 (tf.Tensor): 第一个视图的嵌入，形状为 [batch_size, embedding_dim]。
        z2 (tf.Tensor): 第二个视图的嵌入，形状为 [batch_size, embedding_dim]。
                         z1[i] 和 z2[i] 应构成正样本对。
        temperature (float): 温度超参数，用于缩放相似度。

    Returns:
        tf.Tensor: 一个标量，表示该批次的平均InfoNCE损失。
    """
    # 确保输入是浮点数类型
    z1 = tf.cast(z1, tf.float32)
    z2 = tf.cast(z2, tf.float32)

    # 步骤 1: L2 标准化嵌入向量
    # 这使得点积等价于余弦相似度，并能提高训练稳定性。
    z1_normalized = tf.math.l2_normalize(z1, axis=1)
    z2_normalized = tf.math.l2_normalize(z2, axis=1)

    # 步骤 2: 高效计算成对余弦相似度矩阵
    # 形状: [batch_size, batch_size]
    # similarity_matrix[i, j] 表示 z1[i] 和 z2[j] 之间的相似度
    similarity_matrix = tf.matmul(z1_normalized, z2_normalized, transpose_b=True)

    # 步骤 3: 用温度系数缩放相似度
    similarity_matrix = similarity_matrix / temperature

    # 步骤 4: 创建标签
    # 对于z1中的第i个样本，其正样本是z2中的第i个样本。
    # 因此，标签是 [0, 1, 2, ..., batch_size-1]。
    batch_size = tf.shape(z1)[0]
    labels = tf.range(batch_size)

    # 步骤 5: 使用交叉熵计算损失
    # tf.nn.sparse_softmax_cross_entropy_with_logits 会在内部高效且稳定地
    # 计算 softmax 和 log，避免数值溢出。
    # 它计算的是 -(log(p_positive))，其中p是softmax概率。
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=similarity_matrix
    )
    
    # 步骤 6: 对批次内的所有样本损失取平均
    return tf.reduce_mean(loss)


class SCAGC(tf.keras.Model):

    def __init__(self, X, adj, adj_n, hidden_dim=128, latent_dim=15, dec_dim=None, adj_dim=32):
        super(SCAGC, self).__init__()
        if dec_dim is None:
            dec_dim = [128, 256, 512]
            #dec_dim = [128, 256]
        self.latent_dim = latent_dim
        self.X = X
        self.adj = np.float32(adj)
        self.adj_n = np.float32(adj_n)
        self.n_sample = X.shape[0]
        self.in_dim = X.shape[1]
        self.sparse = False

        initializer = GlorotUniform(seed=7)

        # Encoder
        X_input = Input(shape=self.in_dim)
        h = Dropout(0.2)(X_input)
      
        self.sparse = True
        A_in = Input(shape=self.n_sample, sparse=True)
        h = TAGConv(channels=hidden_dim, kernel_initializer=initializer, activation="relu")([h, A_in])
        z_mean = TAGConv(channels=latent_dim, kernel_initializer=initializer)([h, A_in])

        self.encoder = Model(inputs=[X_input, A_in], outputs=z_mean, name="encoder")
        clustering_layer = ClusteringLayer(name='clustering')(z_mean)
        self.cluster_model = Model(inputs=[X_input, A_in], outputs=clustering_layer, name="cluster_encoder")

        # Adjacency matrix decoder
        
        dec_in = Input(shape=latent_dim)
        h = Dense(units=adj_dim, activation=None)(dec_in)
        h = Bilinear()(h)
        dec_out = Lambda(lambda z: tf.nn.sigmoid(z))(h)
        self.decoderA = Model(inputs=dec_in, outputs=dec_out, name="decoder1")
        

        # Expression matrix decoder

        decx_in = Input(shape=latent_dim)
        h = Dense(units=dec_dim[0], activation="relu")(decx_in)
        h = Dense(units=dec_dim[1], activation="relu")(h)
        h = Dense(units=dec_dim[2], activation="relu")(h)

        pi = Dense(units=self.in_dim, activation='sigmoid', kernel_initializer='glorot_uniform', name='pi')(h)

        disp = Dense(units=self.in_dim, activation=DispAct, kernel_initializer='glorot_uniform', name='dispersion')(h)

        mean = Dense(units=self.in_dim, activation=MeanAct, kernel_initializer='glorot_uniform', name='mean')(h)

        # decx_out = Dense(units=self.in_dim)(h)
        self.decoderX = Model(inputs=decx_in, outputs=[pi, disp, mean], name="decoderX")

    def pre_train(self, epochs=1000, info_step=10, lr=1e-4, W_a=0.3, W_x=1, W_d=0, 
                  contra_temperature=0.7, k=15, gumbel_temperature=1.0, min_dist=0.5, max_dist=20):
      
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if self.sparse == True:
            self.adj_n = tfp.math.dense_to_sparse(self.adj_n)
        adj_2 = self.adj
        adj_n_2 = self.adj_n
        self.adj_history = []
        # Training
        for epoch in range(1, epochs + 1):
            with tf.GradientTape(persistent=True) as tape:
                z = self.encoder([self.X, self.adj_n])
                # X_out = self.decoderX(z)
                pi, disp, mean = self.decoderX(z)
                A_out = self.decoderA(z)
                
                h_norm = tf.linalg.normalize(z, axis=1)[0]
                logits = tf.linalg.matmul(h_norm, h_norm, transpose_b=True)
                # 移除自环
                logits = tf.linalg.set_diag(logits, tf.cast(tf.fill(tf.shape(logits)[0], -np.inf), dtype=logits.dtype))

                # 3. 调用我们的 TF 函数生成可微的邻接矩阵 A_dynamic
                self.k = k
                self.temperature = gumbel_temperature
                a_dynamic = gumbel_topk_sampling_tf(logits, self.k, self.temperature, hard=True)

                # a_dynamic_norm = norm_adj_tf(a_dynamic)
                self.adj = a_dynamic
                dense_adj_n = norm_adj_tf(self.adj) # 得到归一化后的密集张量

                # 关键步骤：将密集张量转换为稀疏张量
                self.adj_n = tf.sparse.from_dense(dense_adj_n)

                z2 = self.encoder([self.X, self.adj_n])
                contrastive_loss = info_nce_loss(z, z2, temperature=contra_temperature)

                if W_d:
                    Dist_loss = tf.reduce_mean(dist_loss(z, min_dist, max_dist=max_dist))
                A_rec_loss = tf.reduce_mean(MSE(self.adj, A_out))
                zinb = ZINB(pi, theta=disp, ridge_lambda=0, debug=False)
                zinb_loss = zinb.loss(self.X, mean, mean=True)
                loss = W_a * A_rec_loss + W_x * zinb_loss + 0.01 * contrastive_loss
                if W_d:
                    loss += W_d * Dist_loss

            vars = self.trainable_weights
            grads = tape.gradient(loss, vars)
            optimizer.apply_gradients(zip(grads, vars))
            if epoch % info_step == 0:
                if W_d:
                    print("Epoch", epoch, " zinb_loss:", zinb_loss.numpy(), "  A_rec_loss:", A_rec_loss.numpy(),
                          "  Dist_loss:", Dist_loss.numpy())
                else:
                    print("Epoch", epoch, " zinb_loss:", zinb_loss.numpy(), "  A_rec_loss:", A_rec_loss.numpy(),
                          "  contrastive_loss:", contrastive_loss.numpy())
                    self.adj_history.append(self.adj.numpy())

        print("Pre_train Finish!")

    def alt_train(self, y, epochs=300, lr=5e-4, W_a=0.3, W_x=1, W_c=1.5, info_step=1, 
                  contra_temperature=0.7, k=15, gumbel_temperature=1.0, n_update=8, centers=None):

        early_stopper = EarlyStopper(patience=10, min_delta=0.001)
        self.cluster_model.get_layer(name='clustering').clusters = centers
        accs = []
        nmis = []
        aris = []

        acc = 0.0
        ari = 0.0
        nmi = 0.0

        # Training
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        for epoch in range(0, epochs):

            if epoch % n_update == 0:
                q = self.cluster_model([self.X, self.adj_n])
                p = self.target_distribution(q)

            with tf.GradientTape(persistent=True) as tape:

                z = self.encoder([self.X, self.adj_n])
                q_out = self.cluster_model([self.X, self.adj_n])
                pi, disp, mean = self.decoderX(z)
                A_out = self.decoderA(z)
                
                h_norm = tf.linalg.normalize(z, axis=1)[0]
                logits = tf.linalg.matmul(h_norm, h_norm, transpose_b=True)
                # 移除自环
                logits = tf.linalg.set_diag(logits, tf.cast(tf.fill(tf.shape(logits)[0], -np.inf), dtype=logits.dtype))

                # 3. 调用我们的 TF 函数生成可微的邻接矩阵 A_dynamic
                self.k = k
                self.temperature = gumbel_temperature
                a_dynamic = gumbel_topk_sampling_tf(logits, self.k, self.temperature, hard=True)

                # 4. (重要) GCNConv 需要归一化的邻接矩阵。Spektral 提供了方便的工具。
                a_dynamic_norm = norm_adj_tf(a_dynamic)
                self.adj = a_dynamic
                dense_adj_n = norm_adj_tf(self.adj) # 得到归一化后的密集张量

                # 关键步骤：将密集张量转换为稀疏张量
                self.adj_n = tf.sparse.from_dense(dense_adj_n)

                z2 = self.encoder([self.X, self.adj_n])
                contrastive_loss = info_nce_loss(z, z2, temperature=contra_temperature)
                
                A_rec_loss = tf.reduce_mean(MSE(self.adj, A_out))
                zinb = ZINB(pi, theta=disp, ridge_lambda=0, debug=False)
                zinb_loss = zinb.loss(self.X, mean, mean=True)
                cluster_loss = tf.reduce_mean(KLD(q_out, p))
                tot_loss = W_a * A_rec_loss + W_x * zinb_loss + W_c * cluster_loss + 0.01 * contrastive_loss

            vars = self.trainable_weights
            grads = tape.gradient(tot_loss, vars)
            optimizer.apply_gradients(zip(grads, vars))

            if epoch % info_step == 0:
                print("Epoch", epoch, " zinb_loss: ", zinb_loss.numpy(), " A_rec_loss: ", A_rec_loss.numpy(),
                      " cluster_loss: ", cluster_loss.numpy())
                y_pred = q.numpy().argmax(1)
                acc = np.round(cluster_acc(y, y_pred), 4)
                nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 4)
                ari = np.round(metrics.adjusted_rand_score(y, y_pred), 4)
                accs.append(acc)
                nmis.append(nmi)
                aris.append(ari)
                self.adj_history.append(self.adj.numpy())

                if early_stopper.early_stop(ari):
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        q_tensor = tf.constant(q)
        y_pred_tensor = np.argmax(q_tensor.numpy(), axis=1)
        self.y_pred = y_pred_tensor

        return self

    def target_distribution(self, q):
        q = q.numpy()
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def embedding(self, count, adj_n):
        return np.array(self.encoder([count, adj_n]))

    def rec_A(self, count, adj_n):
        h = self.encoder([count, adj_n])
        rec_A = self.decoderA(h)
        return np.array(rec_A)

    def get_label(self, count, adj_n):
        if self.sparse:
            adj_n = tfp.math.dense_to_sparse(adj_n)
        clusters = self.cluster_model([count, adj_n]).numpy()
        labels = np.array(clusters.argmax(1))
        return labels.reshape(-1, )
