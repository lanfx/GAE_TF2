from model import GCNModelVAE
from initializations import get_structure, get_random_walk_normalized_laplacian, get_combinatorial_laplacian, \
    get_symmetrical_normalized_laplacian
import collections
import numpy as np
import scipy.sparse as sp
from layer import convert_to_tf_sparse_tensor
import tensorflow as tf

NODE, NODE_NUM, NODE_ELEM_NUM, ELEM_V,A= get_structure(12, 4, 4, 3, 1, 1)
# create features
X = sp.coo_matrix(([-1.], ([192], [2])), shape=(NODE_NUM, 3), dtype='float32')

A_combinatorial = get_combinatorial_laplacian(A)
A_symmetrical = get_symmetrical_normalized_laplacian(A)
A_random_walk = get_random_walk_normalized_laplacian(A)

# creat a namedtuple to save model parameters
ModelInfo = collections.namedtuple('ModelInfo', ['dropout', 'node_num', 'hidden1_output_dim', 'hidden2_output_dim'])
MODEL_INFO = ModelInfo(dropout=0.1, node_num=NODE_NUM, hidden1_output_dim=16, hidden2_output_dim=2)
# create model
A_combinatorial, _ = convert_to_tf_sparse_tensor(A_combinatorial)
model = GCNModelVAE(A_combinatorial, MODEL_INFO)

# 待解决：pos_weight值的设定，yuan
pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()   # val=（邻接矩阵节点的总数-总节点的和）/总结点的和
# norm用于计算最终的cost,cost=norm*tf.reducemin(交叉熵损失(预测值，真实值)) val=总结点数/ （总结点数-总结点和）*2
norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)
# 可以转为静态图进行训练，这样会提高运行速度
def calculate_loss_withKL(original_A,pos_weight,norm,model):
    """

    :param reconstructions: 通过GAE重新生成的新邻接矩阵
    :param original_A: 原始邻接矩阵
    :param pos_weight: weighted_cross_entropy_with_logits参数
    :param model: 论文是假设通过GCN得到的embedding服从正太分布，因此KL散度用来衡量embedding和正态分布的差异
    :return:
    loss:带有KL散度的误差
    accuracy：
    """
    reconstructions_sub = model.reconstructions
    original_A_sub = tf.reshape(original_A.todense(),[-1])
    # calculate loss
    loss = norm * tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(labels=original_A_sub, logits=reconstructions_sub, pos_weight=pos_weight))
    kl = (0.5 / model.node_num) * tf.reduce_mean(
        tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) - tf.square(tf.exp(model.z_log_std)), 1))
    # 与OptimizerAE的区别在于：OptimizerVAE的cost加入了KL损失函数
    loss =loss- kl
    # calculate accracy
    correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(reconstructions_sub), 0.5), tf.int32),
                                tf.cast(original_A_sub, tf.int32))  # tf.equal返回值为Bool
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 将correct_prediction cast成float32类型并求均值
    return loss,accuracy


def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None: # 如过emb为空，则赋值为GCN得到的图的低维表示的分布
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict) # model.z_mean 为要取回的值,赋值为emb,即emb为经过GCN习得的图的低维表示的分布，也就是论文中encoder生成的z
        # sess.run(fetches,feed_dict) fetches为要从计算图中取回的值 feed_dict为输入到计算图中参与计算的值

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T) # rec=reconstructions

    # 存储正例，即原邻接矩阵中存在边的两个顶点在新邻接矩阵的值
    preds = []
    pos = []
    for e in edges_pos:                                 # edges_pos：shape=(263,2) 两列分别代表边的两个顶点i,j，存储的是原邻接矩阵中存在边的两个点——postive
        preds.append(sigmoid(adj_rec[e[0], e[1]]))      # e[0]为顶点i e[1]为顶点j adj_rec[i,j]为新生矩阵中顶点i，j中存在边的情况。preds取值为[0,1]之间的小数，因为经过了sigmoid
        pos.append(adj_orig[e[0], e[1]])                # pos值为0或者1 因为是原邻接矩阵

    # 存储反例，即原邻接矩阵中不存在边的两个顶点在新邻接矩阵的值
    preds_neg = []
    neg = []
    for e in edges_neg:                                 # edges_pos：shape=(忘了,2) 两列分别代表边的两个顶点i,j，存储的是原邻接矩阵中不存在边的两个点——negtive
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    # 进行比对：看一看原邻接矩阵中没边的和有边的点在新邻接矩阵中有没有边
    preds_all = np.hstack([preds, preds_neg])                               # 生成的正样本与负样本即预测值
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))]) # 真实的正样本与负样本即真实值
    roc_score = roc_auc_score(labels_all, preds_all)                        # roc_auc_score为sklearn中方法
    ap_score = average_precision_score(labels_all, preds_all)               # average_precision_score为sklearn中方法

    return roc_score, ap_score


def train(model, features,learning_rate, pos_weight,norm,epochs):
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    trainable_vars = model.trainable_variables
    #cur_loss=0.
    #cur_accuracy=[]
    for epoch in range(1, epochs + 1):
        # 应该拆分训练数据
        with tf.GradientTape() as tape:
            # Forward pass
            model(features) # 返回结果为矩阵经过展平后的一维数组，(105625,)
            # calculate loss
            loss,accuracy = calculate_loss_withKL(A,pos_weight,norm,model)
            # get gradients
        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        #cur_loss += loss
        #cur_accuracy += accuracy
            # cur_preds.append(np.array(preds))
        if (epoch % (epochs // 10)) == 0:
            print(f"Epoch {epoch}/{epochs} -- Loss: {loss: .4f}")
            print(f"cur_accuracy:, {accuracy:.4f}")

train(model=model,features=X,learning_rate=0.1,pos_weight=pos_weight,norm=norm,epochs=200)
reconstructions=model(X)
# get model information
layers=model.layers
variables=model.variables
trainable_vars=model.trainable_variables