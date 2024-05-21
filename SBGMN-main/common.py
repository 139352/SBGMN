DATA_EMB_DIC1 = {
    'bonanza': (7919,1973),   #36543条边
    'review': (182, 304),    # 1170条边
    'ml-1m': (6040, 3952), # 850179条边
}

DATA_EMB_DIC = {**DATA_EMB_DIC1}
for k in DATA_EMB_DIC1:
    for i in range(1, 6):
        DATA_EMB_DIC.update({
            f'{k}-{i}': DATA_EMB_DIC1[k]
        })

import numpy as np
import scipy.sparse as sp
import networkx as nx
from torch.nn import functional as F
import torch
from scipy.spatial.distance import pdist, squareform
import torch.nn as nn

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


def preprocess_features(features):

    rowsum = features.sum(dim=1, keepdim=True)
    normalized_features = features / rowsum

    features = normalized_features.detach().cpu().numpy()
    return sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
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

    adjacency_tensor = torch.tensor(adjacency_matrix.todense())
    row, col = torch.where(adjacency_tensor != 0)

    edge_index = torch.stack([row, col], dim=0)

    return edge_index


def create_graph_from_edge_list(edge_list):

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

def multi_view_fusion(input):

    x = input[:, :, :, 0] + input[:, :, :, 1]

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

def edges_to_adjacency_matrix(num_nodes, edge_list):
    # 创建一个零矩阵作为邻接矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # 将边列表中的连接关系添加到邻接矩阵中
    for edge,value in edge_list.items():
        # edge[0] 和 edge[1] 表示边连接的两个节点
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1  # 无向图需要同时考虑两个方向

    return adjacency_matrix