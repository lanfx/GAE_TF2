import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from initializations import get_GRADN3_info


def sparse_to_tuple(sparse_mx):
    '''
    将矩阵转为coo格式存储，返回值为tuple，封装着稀疏矩阵中非零元素的坐标(coords)，非零元素的值(data)，稀疏矩阵的形状(shape)
    '''
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()  # 将稀疏矩阵转为coo格式存储
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    # np.vstack(A,B) 垂直顺序排列数组
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def dropout_for_tf_sparse_tensor(x, dropout, num_nonzero_elems):
    """Dropout for tf.SparseTensor."""
    noise_shape = [num_nonzero_elems]
    random_tensor = (1 - dropout)
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / (1 - dropout))


def convert_to_tf_sparse_tensor(x):
    '''

    :param x:
    :return: tuple(tf.SparseTensor,int features_nonzero)
    '''
    if not sp.issparse(x):
        x = sp.coo_matrix(x, dtype='float32')
    if x.dtype != 'float32':
        x = x.astype('float32')
    indices = x.nonzero()
    indices = np.vstack((indices[0], indices[1])).transpose()
    values = x.data
    dense_shape = x.shape
    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape), len(values)


def mask_test_edges(adj):
    """

    :param adj:
    :return:
    all_edges [5928,2]
    edges [2964,2]
    test_edges [296,2]
    val_edges [148,2]
    train_edges [2520,2]
    """
    # Function to build test set with 10% positive links

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    # as the adj matrix is symmetry
    # only select test & val data from the upper triangular portion of the adj matrix
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]  # Each edge consists of the id of two vertices [2964,2]


    # get the num of test & validate data
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))


    # get the idx af test & validate data
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    val_edge_idx = all_edge_idx[:num_val]

    # get the data of test & validate edges by id
    test_edges = edges[test_edge_idx]  # return the coords of the edges
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)



    def ismember(a, b, tol=5):
        """
        tol ：
            用于np.round 表示小数部分 如若tol=2 则np.round(0.006)=0.01 np.round(0.005)=0.00
        np.all(a,axis) ：
            延指定轴(axis)进行逻辑AND操作，返回类型为bool类型的tensor；
            如果axis=None则默认对所有维度进行AND,返回结果为单个Bool值
        np.any(a,axis) :
            延指定轴(axis)进行逻辑OR操作，返回类型为bool类型的tensor；
            如果axis=None则默认对所有维度进行OR,返回结果为单个Bool值
        """
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    # 存储测试数据中的反例，反例被定义为在所有边中不存在的边
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):  # test whether the edge [idx_i,idx_j] exist in the total edges
            continue
        if ismember([idx_j, idx_i], edges_all):  # test whether the edge [idx_i,idx_j] exist in the total edges
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        # test_edges_false contains the edges that not exist in the edges_all
        test_edges_false.append([idx_i, idx_j])

    # 存储评估数据中的反例，反例被定义为在所有边中不存在的边
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        # val_edges_false contains the edges that not exist in the train_edges
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)

    # 训练数据与评估数据和测试数据互斥
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)

    # 评估数据与测试数据互斥
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)

    # 为什么要加上转置矩阵？
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


if __name__ == '__main__':
    NODE, A_origin, LOAD, SUPP, NODE_NUM, STRUCTUR_INFO, BARS = get_GRADN3_info('Cantilever')
    #adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false =mask_test_edges(A)
    A = A_origin.todense()
    A_origin = A_origin - sp.dia_matrix((A_origin.diagonal()[np.newaxis, :], [0]), shape=A_origin.shape)  # 将主对角线上的元素变为0
    A_origin.eliminate_zeros()  # adj_orig为稀疏矩阵，删除矩阵中的0元素即存储空间不存储0元素
