import numpy as np
import scipy.sparse as sp
import scipy.io as sio

# def get_structure(node_of_x, node_of_y, node_of_z, length_of_x, length_of_y, length_of_z):
#     [X, Z, Y] = np.meshgrid(np.linspace(0, length_of_x, node_of_x + 1), np.linspace(0, length_of_y, node_of_y + 1),
#                             np.linspace(0, length_of_z, node_of_z + 1))
#     NODE = np.hstack((np.reshape(X, (X.size, 1)), np.reshape(Y, (Y.size, 1)), np.reshape(Z, (Z.size, 1))))
#     NODE_NUM = (node_of_x + 1) * (node_of_y + 1) * (node_of_z + 1)
#     NODE_ELEM_NUM = node_of_x * node_of_y * node_of_z
#     # construct ELEM_V
#     nodeOfElem = np.array([0, node_of_y + 1, node_of_y + 2, 1])
#     aux = np.append([nodeOfElem], [(node_of_x + 1) * (node_of_y + 1) + nodeOfElem])
#     ELEM_V = np.zeros(shape=(NODE_ELEM_NUM, 8), dtype=int)
#     for k in range(0, node_of_z):
#         for j in range(0, node_of_x):
#             for i in range(0, node_of_y):
#                 n = k * node_of_y * node_of_x + j * node_of_y + i
#                 ELEM_V[n] = aux + k * (node_of_y + 1) * (node_of_x + 1) + j * (
#                             node_of_y + 1) + i  # ELEM.V{n}为组成第n个正方体的点的索引的数组,element的顺序为从底至上，按y轴优先顺序存储
#     # construct adj
#     A = np.zeros((NODE_NUM, NODE_NUM), dtype=int)
#     for elem in ELEM_V:
#         for i in elem:
#             for j in elem:
#                 A[i][j] = 1
#     A = sp.coo_matrix(A, dtype='float32')
#     return NODE, NODE_NUM, NODE_ELEM_NUM, ELEM_V,A

def get_GRADN3_info(problme_id):
    GRAND3_DATA = sio.loadmat('./data/CantileverData.mat')
    A=GRAND3_DATA['An'].astype('float32') # 对角为0的矩阵
    NODE=GRAND3_DATA['NODE']
    LOAD=GRAND3_DATA['LOAD']
    LOAD=np.squeeze(LOAD)
    SUPP=GRAND3_DATA['SUPP']
    BARS=GRAND3_DATA['BARS']
    STRUCTUR_INFO=GRAND3_DATA['STRUCTURE_INFO'].flatten()
    NODE_NUM=A.shape[0]
    return NODE,A,LOAD,SUPP,NODE_NUM,STRUCTUR_INFO,BARS

def get_combinatorial_laplacian(A):
    A=A+sp.eye(A.shape[0]) # 论文中的邻接矩阵对角线为1
    rowsum = np.array(A.sum(1))
    D = sp.diags(rowsum.flatten())
    L = D - A
    return L  # ,D


def get_symmetrical_normalized_laplacian(A):
    A = A + sp.eye(A.shape[0])
    rowsum = np.array(A.sum(1))
    D_inv_sqrt = np.power(rowsum, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat_inv_sqrt = sp.diags(D_inv_sqrt)
    # L=A.dot(D_mat_inv_sqrt).transpose().dot(D_mat_inv_sqrt) # D是对角矩阵，与A相乘满足矩阵乘法交换律，但是为什么要将结果转置
    L = D_mat_inv_sqrt.dot(A).dot(D_mat_inv_sqrt)
    return L  # ,D_mat_inv_sqrt


def get_random_walk_normalized_laplacian(A):
    A = A + sp.eye(A.shape[0])
    rowsum = np.array(A.sum(1))
    D_inv = np.power(rowsum, -1.0).flatten()
    D_inv[np.isinf(D_inv)] = 0.
    D_mat_inv = sp.diags(D_inv)
    L = D_mat_inv * A
    return L  # ,D_mat_inv


if __name__ == '__main__':
    NODE,A,LOAD,SUPP,NODE_NUM,STRUCTUR_INFO,BARS=get_GRADN3_info('Cantilever')
    load_val = np.array(LOAD[1:4])

    # X = sp.coo_matrix(([-1], ([192], [2])), shape=(NODE_NUM, 3))
    #
    # A_combinatorial = get_combinatorial_laplacian(A)
    # A_symmetrical = get_symmetrical_normalized_laplacian(A)
    # A_random_walk = get_random_walk_normalized_laplacian(A)
    #
    # result1 = A_combinatorial * X
    # result2 = A_symmetrical * X
    # result3 = A_random_walk * X
