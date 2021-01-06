from layer import convert_to_tf_sparse_tensor
from model import GCNModelVAE
from initializations import get_GRADN3_info, get_random_walk_normalized_laplacian, get_combinatorial_laplacian, get_symmetrical_normalized_laplacian
from preproposing import mask_test_edges,convert_to_tf_sparse_tensor
from optimizer import calculate_loss_withKL, get_roc_score
import collections
import numpy as np
import scipy.sparse as sp

import tensorflow as tf

# Load data
NODE, A, LOAD, SUPP, NODE_NUM, STRUCTUR_INFO, BARS = get_GRADN3_info('Cantilever')

# create features 原GAE是将所有点所属的类别矩阵(one hot)作为特征即每一行为一个点，每一列
load_num_id = LOAD[0]
load_val = LOAD[1:4]
load_index, load_val_atindex = np.squeeze([[index, val] for index, val in enumerate(load_val) if val != 0])
#X = sp.coo_matrix(([load_val_atindex], ([load_num_id], [load_index])), shape=(NODE_NUM, 3), dtype='float32')
X = sp.eye(NODE_NUM).tocoo()
# 对原始矩阵进行处理
A_origin = A
A_origin = A_origin - sp.dia_matrix((A_origin.diagonal()[np.newaxis, :], [0]), shape=A_origin.shape) # 将主对角线上的元素变为0
A_origin.eliminate_zeros() # adj_orig为稀疏矩阵，删除矩阵中的0元素即存储空间不存储0元素

# 构造训练，验证，测试集
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(A)
A = adj_train # adj_train 为对角线为0，去除原图中一部分链接后的邻接矩阵

# 构建模型
ModelInfo = collections.namedtuple('ModelInfo', ['dropout', 'node_num', 'hidden1_output_dim', 'hidden2_output_dim'])
MODEL_INFO = ModelInfo(dropout=0.,
                       node_num=NODE_NUM,
                       hidden1_output_dim=32,
                       hidden2_output_dim=16)

A_combinatorial = get_combinatorial_laplacian(A)
A_symmetrical = get_symmetrical_normalized_laplacian(A)
A_random_walk = get_random_walk_normalized_laplacian(A)

A_symmetrical, _ = convert_to_tf_sparse_tensor(A_symmetrical)

model = GCNModelVAE(A_symmetrical, MODEL_INFO)

# 待解决：pos_weight值的设定
pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()  # val=（邻接矩阵节点的总数-总节点的和）/总结点的和
# norm用于计算最终的cost,cost=norm*tf.reducemin(交叉熵损失(预测值，真实值)) val=总结点数/ （总结点数-总结点和）*2
norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)

adj_label = adj_train + sp.eye(adj_train.shape[0]) # 在adj_train基础上将对角线元素变为1参与模型的训练
adj_label, _= convert_to_tf_sparse_tensor(adj_label)


def train(model, features, learning_rate, pos_weight, norm, epochs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    trainable_vars = model.trainable_variables
    # cur_loss=0.
    # cur_accuracy=[]
    for epoch in range(1, epochs + 1):
        with tf.GradientTape() as tape:
            # Forward pass
            model(features)
            # calculate loss
            loss, accuracy = calculate_loss_withKL(labels=tf.reshape(tf.sparse.to_dense(adj_label), [-1]),
                                                   pos_weight=pos_weight,
                                                   norm=norm,
                                                   model=model)

        val_roc_score= []
        # calculate gradients
        gradients = tape.gradient(loss, trainable_vars)
        # optimize the variavle
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        roc_curr, ap_curr = get_roc_score(A_origin, model.z_mean, val_edges, val_edges_false)
        val_roc_score.append(roc_curr)
        # cur_loss += loss
        # cur_accuracy += accuracy
        # cur_preds.append(np.array(preds))
        if (epoch % (epochs // 10)) == 0:
            print("Epoch {}/{}".format(epoch,epochs),
                  " -- Loss: {:.4f}".format(loss),
                  "cur_accuracy: {:.4f}".format(accuracy),
                  "val_roc: {:.5f}".format(val_roc_score[-1]),
                  "val_ap: {:.5f}".format(ap_curr))


train(model=model,
      features=X,
      learning_rate=0.01,
      pos_weight=pos_weight,
      norm=norm,
      epochs=200)

roc_score, ap_score = get_roc_score(A_origin, model.z, test_edges, test_edges_false)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
# get model information
layers = model.layers
variables = model.variables
trainable_vars = model.trainable_variables
