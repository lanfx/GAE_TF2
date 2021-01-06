import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from preproposing import dropout_for_tf_sparse_tensor, convert_to_tf_sparse_tensor

class GraphConvolutionSparse(tf.keras.layers.Layer):
    """Graph convolution layer for sparse inputs(features)."""

    def __init__(self,
                 output_dim,
                 adj,
                 dropout=0.,
                 activation=tf.nn.relu,
                 name=None):
        super(GraphConvolutionSparse, self).__init__(name=name)
        self.output_dim = output_dim
        self.adj = adj
        self.dropout = dropout
        self.activation = activation

    def build(self, inputs_shape):
        initializer_glorot = tf.keras.initializers.GlorotNormal()
        initializer_one = tf.ones
        self.weight = tf.Variable(initializer_glorot(shape=(inputs_shape[-1], self.output_dim)),
                                  name='weight')

    def call(self, inputs):
        '''
        Graph convolution layer for sparse inputs(features)
        :param inputs: features,暂为shape为(nodenum,3)的稀疏矩阵，3列分别代表该节点在x,y,z方向上的受力情况，非0值即代表该点受力
        :return:
        outputs： dense matrix
        '''
        x = inputs
        x, features_nonzero = convert_to_tf_sparse_tensor(x)
        x = dropout_for_tf_sparse_tensor(x, self.dropout, features_nonzero)
        x_mul_weight = tf.sparse.sparse_dense_matmul(x, self.weight)
        adj_mul_x_mul_weight = tf.sparse.sparse_dense_matmul(self.adj,
                                                             x_mul_weight)
        outputs = self.activation(adj_mul_x_mul_weight)
        return outputs


class GraphConvolution(tf.keras.layers.Layer):
    """Graph convolution layer for normal inputs(features)."""

    def __init__(self,
                 output_dim,
                 adj,
                 dropout=0.,
                 activation=tf.nn.relu,
                 name=None):
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

    def __init__(self,
                 dropout=0.,
                 activation=tf.nn.sigmoid,
                 name=None):
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
