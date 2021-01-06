import tensorflow as tf
from layer import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, convert_to_tf_sparse_tensor
import collections


class GCNModelVAE(tf.keras.Model):

    def __init__(self, adj, MODEL_INFO, name=None, **kwargs):
        super(GCNModelVAE, self).__init__(name=name)

        self.node_num = MODEL_INFO.node_num
        self.output_dim = MODEL_INFO.hidden2_output_dim

        # input feature is sparse
        self.embedding_sparse = GraphConvolutionSparse(
            output_dim=MODEL_INFO.hidden1_output_dim,
            adj=adj,
            dropout=MODEL_INFO.dropout,
            activation=tf.nn.relu
        )

        # encoder1:self.z_mean 用于习得模型的均值
        self.encoder_z_mean = GraphConvolution(
            output_dim=MODEL_INFO.hidden2_output_dim,
            adj=adj,
            dropout=MODEL_INFO.dropout,
            activation=lambda x: x
        )

        # encoder2:self.z_log_std用于习得模型的标准差:log 标准差=GCN(X,A),标准差=e^(GCN(X,A))
        # encoder1,encoder2共享参数
        self.encoder_z_log_std = GraphConvolution(
            output_dim=MODEL_INFO.hidden2_output_dim,
            adj=adj,
            dropout=MODEL_INFO.dropout,
            activation=lambda x: x
        )

        # decoder
        self.decoder = InnerProductDecoder(
            dropout=MODEL_INFO.dropout,
            activation=lambda x: x)

    def call(self, inputs):
        # Embedding
        self.embedding = self.embedding_sparse(inputs)

        # Encoder
        self.z_mean = self.encoder_z_mean(self.embedding)
        self.z_log_std = self.encoder_z_log_std(self.embedding)

        self.z = self.z_mean + tf.random.normal([self.node_num, self.output_dim]) * tf.exp(self.z_log_std)

        # Decoder
        self.reconstructions = self.decoder(self.z)
        return self.reconstructions

if __name__=='__main__':
    pass
# ModelInfo=collections.namedtuple('ModelInfo',['dropout','node_num','hidden1_output_dim','hidden2_output_dim'])
# MODEL_INFO=ModelInfo(dropout=0.1,node_num=4,hidden1_output_dim=4,hidden2_output_dim=2)
# print(MODEL_INFO.dropout)
# adj=tf.random.normal((4,4))
# model=GCNModelVAE(adj,MODEL_INFO)
# X=tf.ones([4,2])
# result=model(X)
# embedding,z_mean,z_log_std=model(X)
# result=tf.random.normal([MODEL_INFO.node_num, MODEL_INFO.hidden2_output_dim])
