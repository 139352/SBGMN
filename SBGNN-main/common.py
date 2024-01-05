DATA_EMB_DIC1 = {
    'bonanza': (7919,1973),   #36543条边
    'house1to10': (515, 1281),   #114378条边
    'senate1to10': (145, 1056),   #27083条边
    'review': (182, 304),# 153k, 1G   1170条边
    'ml-1m': (6040, 3952), # 23m870k, 28G  850179条边
    'amazon-book': (35736, 38121)
}

DATA_EMB_DIC = {**DATA_EMB_DIC1}
for k in DATA_EMB_DIC1:
    for i in range(1, 6):
        DATA_EMB_DIC.update({
            f'{k}-{i}': DATA_EMB_DIC1[k]
        })

import numpy as np
# import tensorflow as tf
import scipy.sparse as sp
import networkx as nx
from torch.nn import functional as F
import torch
from scipy.spatial.distance import pdist, squareform
import torch.nn as nn
# from keras import backend as K
# from keras.layers import Permute, multiply, Dense, Reshape, Lambda

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eps = 1e-8
def get_center_similarity(out1, out2, A):
    out1 = F.normalize(out1.detach()).to(device)
    out2 = F.normalize(out2.detach()).to(device)

    s1 = torch.matmul(out1, out1.T).to(device)
    s2 = torch.matmul(out2, out2.T).to(device)

    sim1 = (s1 * A).sum(dim=1, keepdim=True) / (A.sum(dim=1, keepdim=True) + eps).to(device)  # Structure-level Knowledge Distillation.
    sim2 = (s2 * A).sum(dim=1, keepdim=True) / (A.sum(dim=1, keepdim=True) + eps).to(device)

    return sim1, sim2


def get_structure_loss(out1, out2, A):
    out1 = F.normalize(out1)
    out2 = F.normalize(out2)

    s1 = F.log_softmax(torch.matmul(out1, out1.T) * A, dim=1)
    s2 = F.softmax(torch.matmul(out2, out2.T) * A, dim=1)

    struct_loss = F.kl_div(s1, s2, reduction='none').sum(dim=1, keepdim=True)

    return struct_loss


def preprocess_features(features):   #用于预处理特征矩阵的函数。它的主要功能包括将特征矩阵进行行归一化（row-normalize）并将其转换成元组表示。
    """Row-normalize feature matrix and convert to tuple representation"""
    # rowsum = np.array(features.sum(1))
    # r_inv = np.power(rowsum, -1).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)
    # features = r_mat_inv.dot(features)
    # return sparse_to_tuple(features)
    rowsum = features.sum(dim=1, keepdim=True)
    normalized_features = features / rowsum

    # 将 PyTorch Tensor 转换为 NumPy 数组，不影响梯度信息
    features = normalized_features.detach().cpu().numpy()

    # 将 PyTorch Tensor 转换为 NumPy 数组
    # features = normalized_features.numpy()
    return sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        # if not sp.isspmatrix_coo(mx):
        #     mx = mx.tocoo()
        # coords = np.vstack((mx.row, mx.col)).transpose()
        # values = mx.data
        # shape = mx.shape
        coo = sp.coo_matrix(mx)
        coords = np.vstack((coo.row, coo.col)).transpose()
        values = coo.data
        shape = coo.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def adjacency_matrix_to_edge_index(adjacency_matrix):
    # 找到邻接矩阵中非零元素的坐标
    adjacency_tensor = torch.tensor(adjacency_matrix.todense())
    row, col = torch.where(adjacency_tensor != 0)
    # 将坐标信息表示为边的索引
    edge_index = torch.stack([row, col], dim=0)

    return edge_index


def create_graph_from_edge_list(edge_list):
    """
    从边列表创建图
    """
    G = nx.Graph()
    i = 0
    for edge_array in edge_list:
        if edge_array.shape[0]==0:
            G.add_edge(i, i)
            i = i + 1
        for edge in edge_array:
            edge_tuple = tuple(edge.tolist())
            G.add_edge(*edge_tuple)
            i = i + 1


    return G


def weisfeiler_lehman_labels_from_edge_list(edge_list, max_iterations, node_num):
    """
    从边列表计算 Weisfeiler-Lehman 标签
    """

    # 创建一个空的二部图
    # G = nx.DiGraph()

    # 添加边到二部图中
    # G.add_edges_from(edge_list)
    G = create_graph_from_edge_list(edge_list)

    labels = {}
    for node in G.nodes():
        # labels[node] = str(np.random.randint(0, 2))  # 这里使用随机生成的标签，实际中应根据你的需求设置节点标签
        labels[node] = '1'

    for iteration in range(max_iterations):
        new_labels = {}
        for node in G.nodes():
            neighbors = sorted(G.neighbors(node))   #通过 sorted() 函数对这个迭代器进行排序，将邻居节点按照节点编号从小到大排列。排序后的结果将存储在 neighbors 变量中
            neighbor_labels = [labels[neighbor] for neighbor in neighbors]
            label_string = labels[node] + ''.join(neighbor_labels)
            new_labels[node] = str(hash(label_string))[:4]

        labels = new_labels

    return labels

# def weisfeiler_lehman_labels_from_edge_list(num_nodes, edges):
#     edge_index = edges
#     # num_edges = edge_index.shape[0]
#     ones_like_tensor = torch.ones(num_nodes)
#     labels = torch.ones_like(ones_like_tensor)
#
#     for iteration in range(2):  # 迭代两次，可以根据需要调整迭代次数
#         new_labels = labels.clone()
#         for node in range(num_nodes):
#             # 获取邻居节点的标签并排序
#             neighbors = edge_index[1, edge_index[0] == node]
#             neighbor_labels = labels[neighbors].sort().values
#
#             # 连接邻居节点的标签成字符串
#             neighbor_labels_str = ''.join(map(str, neighbor_labels.tolist()))
#
#             # 使用哈希函数映射字符串为新标签
#             new_label = hash(neighbor_labels_str)
#
#             # 更新节点的标签
#             new_labels[node] = new_label
#
#         # 更新标签
#         labels = new_labels
#
#     return labels



def generate_new_adj(X, delta):

    # temp_x = X.toarray()
    temp_x = X.copy()
    similarity = 1 - pdist(temp_x, 'cosine')
    where_are_nan = np.isnan(similarity)
    similarity[where_are_nan] = 0
    similarity[similarity > delta] = 1
    similarity[similarity <= delta] = 0
    A = squareform(similarity)
    adj = sp.csr_matrix(A)

    return adj


class STDLayerNorm(nn.Module):
    """Construct a layernorm module"""
    def __init__(self, num_features: int, eps=1e-6):
        super(STDLayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        #mean = x.mean(axis=-1)
        x=torch.tensor(x).to(device)
        mean=torch.mean(x,dim=-1,keepdim=True)
        std=torch.std(x,dim=-1,keepdim=True)
        #std = x.std(axis=-1, keep_dims=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# def expend_tensor_dim_l(H):
#     expend_x = K.expand_dims(H, 0)
#     return expend_x


# def squeeze_tensor_dim_l(H):
#     squeeze_x = K.squeeze(H, 0)
#     return squeeze_x


def multi_view_fusion(input):

    x = input[:, :, :, 0] + input[:, :, :, 1]

    return x


class GlobalAveragePooling2D(torch.nn.Module):
    def forward(self, x):
        # 使用 adaptive_avg_pool2d 进行全局平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # 将输出张量的形状调整为 (batch_size, num_channels)
        # x = x.view(x.size(0), -1)
        return x

class SimpleLinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLinearLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# def view_attention_block(input, view_num):
#     # input[:, :, :, 0] = K.dot(adj, input[:, :, :, 0])
#     # input[:, :, :, 1] = K.dot(adj2, input[:, :, :, 1])
#     init = input
#     input_shape = (1, 1, view_num)
#
#     global_avg_pooling = GlobalAveragePooling2D()
#     weight = global_avg_pooling(init)
#     # weight =  weight.view(*input_shape)
#     # weight = Reshape(input_shape)(init)
#     dense_layer1 = SimpleLinearLayer(weight.shape[1], 6)
#     dense_layer2 = SimpleLinearLayer(6, 3)
#     dense_layer3 = SimpleLinearLayer(3, 2)
#     weight = torch.FloatTensor(weight)
#     output = dense_layer1(weight)
#     output = dense_layer2(output)
#     output = dense_layer3(output)
#     weight = Dense(5, activation='sigmoid', kernel_initializer='glorot_uniform', use_bias=True)(weight)
#     # weight = Dense(3, activation='sigmoid', kernel_initializer='glorot_uniform', use_bias=True)(weight)
#     # weight = Dense(view_num, activation='softmax', kernel_initializer='glorot_uniform', use_bias=True)(weight)
#
#     # if K.image_data_format() == 'channels_first':
#     #     weight = Permute((3, 1, 2))(weight)
#     output = output.permute(0, 2, 3, 1)
#
#     # temp_x = multiply([init, weight])
#     temp_x = torch.mul(init, output)
#
#     x = multi_view_fusion(temp_x)
#
#     return x

# def glorot(shape, name=None):
#     """Glorot & Bengio (AISTATS 2010) init."""
#     init_range = np.sqrt(6.0/(shape[0]+shape[1]))
#     initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
#     return tf.Variable(initial, name=name)

# def zeros(shape, name=None):
#     """All zeros."""
#     initial = tf.zeros(shape, dtype=tf.float32)
#     return tf.Variable(initial, name=name)

def edges_to_adjacency_matrix(num_nodes, edge_list):
    # 创建一个零矩阵作为邻接矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # 将边列表中的连接关系添加到邻接矩阵中
    for edge,value in edge_list.items():
        # edge[0] 和 edge[1] 表示边连接的两个节点
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1  # 无向图需要同时考虑两个方向

    return adjacency_matrix