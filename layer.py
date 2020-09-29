import tensorflow as tf
import numpy as np
import scipy.sparse as sp


def dropout_for_tf_sparse_tensor(x, dropout, num_nonzero_elems):
    """Dropout for tf.SparseTensor."""
    noise_shape = [num_nonzero_elems]
    random_tensor = (1 - dropout)
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / (1 - dropout))


def convert_to_tf_sparse_tensor(x):
    """
    将稀疏矩阵转为coo存储，再转为tensorflow.sparse.SparseTensor
    :param sparse_tensor:
    :return: (tf.SparseTensor,int features_nonzero)
    """
    if not sp.issparse(x):
        x = sp.coo_matrix(x, dtype='float32')
    indices = x.nonzero()
    indices = np.vstack((indices[0], indices[1])).transpose()
    values = x.data
    dense_shape = x.shape
    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape), len(values)


class GraphConvolutionSparse(tf.keras.layers.Layer):
    """Graph convolution layer for sparse inputs(features)."""

    def __init__(self, output_dim, adj, dropout=0., activation=tf.nn.relu, name=None):
        super(GraphConvolutionSparse, self).__init__(name=name)
        self.output_dim = output_dim
        self.adj = adj
        self.dropout = dropout
        self.activation = activation

    def build(self, inputs_shape):
        initializer_glorot = tf.keras.initializers.GlorotNormal()
        initializer_one = tf.ones
        self.weight = tf.Variable(
            initializer_glorot(shape=(inputs_shape[-1], self.output_dim)), name='weight')

    def call(self, inputs):
        x = inputs
        x, features_nonzero = convert_to_tf_sparse_tensor(x)
        x = dropout_for_tf_sparse_tensor(x, self.dropout, features_nonzero)
        x_mul_weight = tf.sparse.sparse_dense_matmul(x, self.weight)
        adj_mul_x_mul_weight = tf.sparse.sparse_dense_matmul(self.adj,
                                                             x_mul_weight)  # sparse_dense_matmul的接收的稀疏矩阵类型必须为tf.sparse
        outputs = self.activation(adj_mul_x_mul_weight)
        return outputs


class GraphConvolution(tf.keras.layers.Layer):
    """Graph convolution layer for normal inputs(features)."""

    def __init__(self, output_dim, adj, dropout=0., activation=tf.nn.relu, name=None):
        super(GraphConvolution, self).__init__(name=name)
        self.output_dim = output_dim
        self.dropout = dropout
        self.adj = adj
        self.activation = activation

    def build(self, inputs_shape):
        initializer = tf.keras.initializers.GlorotNormal()
        self.weight = tf.Variable(
            initializer(shape=(inputs_shape[-1], self.output_dim)), name='weight')

    def call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, self.dropout)
        x_mul_weight = tf.matmul(x, self.weight)
        adj_mul_x_mul_weight = tf.sparse.sparse_dense_matmul(self.adj, x_mul_weight)
        outputs = self.activation(adj_mul_x_mul_weight)
        return outputs


class InnerProductDecoder(tf.keras.layers.Layer):
    """Decoder model layer for link prediction."""

    def __init__(self, dropout=0., activation=tf.nn.sigmoid, name=None):
        super(InnerProductDecoder, self).__init__(name=None)
        self.dropout = dropout
        self.activation = activation

    def call(self, inputs):
        inputs = tf.nn.dropout(inputs, self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])  # -1相当于flaetten
        outputs = self.activation(x)
        return outputs


if __name__ == '__main__':
    adj, _ = convert_to_tf_sparse_tensor(tf.ones((4, 4)))
    x = tf.eye(4)
    gcs = GraphConvolutionSparse(output_dim=2, adj=adj, dropout=0.1)
    result = gcs(x)
