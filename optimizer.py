import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# 可以转为静态图进行训练，这样会提高运行速度
def calculate_loss_withKL(labels, pos_weight, norm, model):
    """
    将通过GAE预测的邻接矩阵同原始邻接矩阵进行损失计算
    :param labels: 展平后的原始邻接矩阵
    :param pos_weight: weighted_cross_entropy_with_logits参数
    :param model: 论文是假设通过GCN得到的embedding服从正太分布，因此KL散度用来衡量embedding和正态分布的差异
    :return:
    loss:带有KL散度的误差
    accuracy：正确率
    """
    preds_sub = model.reconstructions
    labels_sub = labels

    # calculate cross entropy loss
    loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels_sub,
                                                                          logits=preds_sub,
                                                                          pos_weight=pos_weight))
    # calculate the kl
    kl = (0.5 / model.node_num) * tf.reduce_mean(
        tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) - tf.square(tf.exp(model.z_log_std)), 1))

    # calculate the loss
    loss = loss - kl

    # calculate accuracy
    correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                  tf.cast(labels_sub, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, accuracy


def get_roc_score(adj_orig, emb, edges_pos, edges_neg, ):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, tf.transpose(emb))  # rec=reconstructions

    # 存储正例，即原邻接矩阵中存在边的两个顶点在新邻接矩阵的值
    preds = []
    pos = []
    for e in edges_pos:  # edges_pos：shape=(263,2) 两列分别代表边的两个顶点i,j，存储的是原邻接矩阵中存在边的两个点——postive
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
            #adj_rec[e[0], e[1]] 为decoder生成的表示两节点间是否存在连接关系的score，经过sigmoid转为概率。为二分类问题，要么有边，要么每边。
        pos.append(adj_orig[e[0], e[1]])

    # 存储反例，即原邻接矩阵中不存在边的两个顶点在新邻接矩阵的值
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    # 进行比对：看一看原邻接矩阵中没边的和有边的点在新邻接矩阵中有没有边
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))]) # 水平堆叠
    # 进行roc比较的是边而不是矩阵
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

if __name__ == '__main__':
    A= tf.random.normal([3,3])
    B= tf.transpose(A)
