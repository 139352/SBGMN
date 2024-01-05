#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author: huangjunjie
@file: sbgnn.py
@time: 2021/03/28
"""


import os
import sys
import time
import random
import argparse
import subprocess

from collections import defaultdict
from scipy.sparse import lil_matrix

import numpy as np
import networkx as nx
from ppo_net import PPO
from common import get_center_similarity, get_structure_loss, preprocess_features, adjacency_matrix_to_edge_index
from common import weisfeiler_lehman_labels_from_edge_list, generate_new_adj, expend_tensor_dim_l, view_attention_block
from common import squeeze_tensor_dim_l, edges_to_adjacency_matrix

import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch.distributions import Categorical
import torch.optim as optim


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score


from tqdm import tqdm

import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import logging
# https://docs.python.org/3/howto/logging.html#logging-advanced-tutorial


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dirpath', default=BASE_DIR, help='Current Dir')
parser.add_argument('--device', type=str, default='cpu', help='Devices')
parser.add_argument('--dataset_name', type=str, default='house1to10-1')
parser.add_argument('--a_emb_size', type=int, default=32, help='Embeding A Size')
parser.add_argument('--b_emb_size', type=int, default=32, help='Embeding B Size')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight Decay')
parser.add_argument('--lr', type=float, default=0.05, help='Learning Rate')#0.005
parser.add_argument('--seed', type=int, default=13, help='Random seed')
parser.add_argument('--epoch', type=int, default=2000, help='Epoch')
parser.add_argument('--gnn_layer_num', type=int, default=2, help='GNN Layer')
parser.add_argument('--batch_size', type=int, default=500, help='Batch Size')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
parser.add_argument('--agg', type=str, default='AttentionAggregator', choices=['AttentionAggregator', 'MeanAggregator'], help='Aggregator')
args = parser.parse_args()


# TODO

exclude_hyper_params = ['dirpath', 'device']
hyper_params = dict(vars(args))
for exclude_p in exclude_hyper_params:    #将 hyper_params 字典中在 exclude_hyper_params 列表中指定的键排除（删除）。这可能是为了在调整超参数时排除某些特定的超参数设置
    del hyper_params[exclude_p]

hyper_params = "~".join([f"{k}-{v}" for k,v in hyper_params.items()])

from torch.utils.tensorboard import SummaryWriter
# https://pytorch.org/docs/stable/tensorboard.html
tb_writer = SummaryWriter(comment=hyper_params)


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
# # setup seed
# setup_seed(args.seed)

from common import DATA_EMB_DIC

# args.device = 'cpu'
args.device = torch.device(args.device)

class GraphSAGE_NET(torch.nn.Module):

    def __init__(self, input_dim, hidden, classes):
        super(GraphSAGE_NET, self).__init__()
        self.sage1 = SAGEConv(input_dim, hidden)  # 定义两层GraphSAGE层
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, x, edge_index):

        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)

        return x


class GAT_NET(torch.nn.Module):   #多跳的GAT
    def __init__(self, input_dim, hidden, classes, heads=4, num_hops=2):
        super(GAT_NET, self).__init__()
        self.gat_1 = GATConv(input_dim, hidden)
        self.gat_layers = torch.nn.ModuleList([
            GATConv(hidden, hidden, heads=heads) for _ in range(num_hops-1)])  # 定义GAT层，使用多头注意力机制
        self.final_layer = GATConv(hidden, classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。

    def forward(self, x, edge_index1):
        # 通过多跳邻居信息的聚合
        x = self.gat_1(x, edge_index1)
        # for gat_layer in self.gat_layers:
        #     # 传入卷积层
        #     x = gat_layer(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.final_layer(x, edge_index1)
        return x


class MeanAggregator(nn.Module):
    def __init__(self, a_dim, b_dim):
        super(MeanAggregator, self).__init__()

        self.out_mlp_layer = nn.Sequential(
            nn.Linear(a_dim, b_dim)
        )

    def forward(self, edge_dic_list: dict, node_num):
        edges1 = []
        for node in range(node_num):
            neighs = np.array(edge_dic_list[node]).reshape(-1, 1)
            a = np.array([node]).repeat(len(neighs)).reshape(-1, 1)
            edges1.append(np.concatenate([a, neighs], axis=1))

        # edges2 = []
        # for node in range(node_num_b):
        #     neighs = np.array(edge_dic_list[node]).reshape(-1, 1)
        #     a = np.array([node]).repeat(len(neighs)).reshape(-1, 1)
        #     edges2.append(np.concatenate([a, neighs], axis=1))

        # 计算 WL 标签
        max_iterations = 3  # 设置迭代次数
        output_emb_a = weisfeiler_lehman_labels_from_edge_list(edges1, max_iterations, node_num)
        # output_emb = self.out_mlp_layer(output_emb)

        # max_iterations = 3  # 设置迭代次数
        # output_emb_b = weisfeiler_lehman_labels_from_edge_list(edges2, max_iterations)

        # output_emb = torch.tensor(output_emb['data'])
        # output_emb_a =

        return output_emb_a

class SBGNNLayer(nn.Module):
    def __init__(self, edgelist_pos, edgelist_neg, dataset_name=args.dataset_name, emb_size_a=32, emb_size_b=32, aggregator=MeanAggregator):
        super(SBGNNLayer, self).__init__()
        #
        self.set_a_num, self.set_b_num = DATA_EMB_DIC[dataset_name]

        # self.feature_a = feature_a
        # self.feature_b = feature_b
        self.edgelist_pos = edgelist_pos
        self.edgelist_neg = edgelist_neg

        self.agg_pos = aggregator(emb_size_a, emb_size_b)
        self.agg_neg = aggregator(emb_size_a, emb_size_b)

        self.update_func = nn.Sequential(        #self.update_func 将会是一个由以下几个层组成的神经网络模型。这个模型可能被用于在图神经网络中进行节点特征的更新或聚合操作
            nn.Dropout(args.dropout),
            nn.Linear(emb_size_a * 5, emb_size_a * 2),
            nn.PReLU(),
            nn.Linear(emb_size_b * 2, emb_size_b)

        )

    def forward(self, feature_a, feature_b):
        # assert feature_a.size()[0] == self.set_a_num, 'set_b_num error'
        # assert feature_b.size()[0] == self.set_b_num, 'set_b_num error'
        feature_dim_a = feature_a.shape[1]
        feature_dim_b = feature_b.shape[1]

        node_num_a, node_num_b = self.set_a_num, self.set_b_num
        node_num = node_num_a + node_num_b

        m_all_pos = self.agg_pos(self.edgelist_pos, node_num)
        sorted_keys_pos = sorted(m_all_pos.keys())
        tensor_pos = torch.tensor([float(m_all_pos[key]) for key in sorted_keys_pos], dtype=torch.float).view(-1, 1)
        # tensor_pos = self.update_func(tensor_pos)
        # sorted_keys_pos_b = sorted(m_a_from_b_pos.keys())
        # tensor_pos_b = torch.tensor([float(m_b_from_a_pos[key]) for key in sorted_keys_pos_b], dtype=torch.float).view(-1,
        #                                                                                                            1)
        m_all_neg = self.agg_neg(self.edgelist_neg, node_num)
        sorted_keys_neg_a = sorted(m_all_neg.keys())
        tensor_neg = torch.tensor([float(m_all_neg[key]) for key in sorted_keys_neg_a], dtype=torch.float).view(-1, 1)
        # tensor_neg = self.update_func(tensor_neg)
        # sorted_keys_neg_b = sorted(m_b_from_a_neg.keys())
        # tensor_neg_b = torch.tensor([float(m_b_from_a_neg[key]) for key in sorted_keys_neg_b], dtype=torch.float).view(-1,
        #                                                                                                              1)

        new_feature = torch.cat([feature_a, feature_b], dim=0)

        new_feature = torch.cat([new_feature, tensor_pos, tensor_neg], dim=1)
        # new_feature_b = torch.cat([feature_b, tensor_neg], dim=1)

        # new_feature = self.update_func(new_feature)

        return new_feature

def similarity_loss(features_i, features_j):
    return 0.5 * torch.norm(features_i - features_j, p=2) ** 2

        # 定义不同性损失
def dissimilarity_loss(features_i, features_j, margin=1.0):
    distance = torch.norm(features_i - features_j, p=2) ** 2
    return 0.5 * torch.max(torch.tensor(0.0), margin - distance)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)

        pt = torch.exp(-BCE_loss).requires_grad_()
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class SBGNN(nn.Module):
    def __init__(self, edgelists, trainedgelist,
                    dataset_name=args.dataset_name, layer_num=1, emb_size_a=32, emb_size_b=32, aggregator1=MeanAggregator):
        super(SBGNN, self).__init__()

        # assert edgelists must compelte
        assert len(edgelists) == 2, 'must 2 edgelists'   #assert len(edgelists) == 8 表示断言条件，要求列表 edgelists 的长度必须等于 8，'must 8 edgelists' 是在触发异常时显示的错误消息，它会在 AssertionError 异常的提示信息中显示出来
        # edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,\
        #             edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg = edgelists    #反斜杠（\）用于表示代码的换行符，它将一行代码分成两行来书写，但在实际执行时会被视为一行代码
        edgelist_pos, edgelist_neg = edgelists
        self.update_func = nn.Sequential(  # self.update_func 将会是一个由以下几个层组成的神经网络模型。这个模型可能被用于在图神经网络中进行节点特征的更新或聚合操作
            nn.Dropout(args.dropout),
            # nn.Linear(emb_size_a+2 , emb_size_a * 2),
            nn.PReLU(),
            nn.Linear(emb_size_b * 2, emb_size_b)

        )

        self.multi_view_model = GraphSAGE_NET(input_dim=34, hidden=64, classes=34)
        self.multi_gnn_model1 = GAT_NET(input_dim=34, hidden=16, classes=32, heads=4, num_hops=2)
        self.multi_gnn_model2 = GAT_NET(input_dim=34, hidden=16, classes=32, heads=4, num_hops=2)
        ## self.classifier = SimpleClassifier(input_size=64, output_size=1)
        self.classifier = SimpleClassifier(input_size=32, output_size=1)

        self.trainedgelist = trainedgelist
        self.featureless = False
        self.set_a_num, self.set_b_num = DATA_EMB_DIC[dataset_name]    #读取到a和b类节点的个数分别是多少

        self.act = tf.nn.relu
        self.edge_pos = edgelist_pos

        self.features_a = nn.Embedding(self.set_a_num, emb_size_a)   # self.features_a 是一个嵌入层，用于将节点A的整数索引映射为具有 emb_size_a 维度的密集向量表示。这种嵌入层通常用于学习节点的低维度表示
        self.features_b = nn.Embedding(self.set_b_num, emb_size_b)
        self.features_a.weight.requires_grad = True
        self.features_b.weight.requires_grad = True
        # features_a = features_a.to(args.device)
        # features_b = features_b.to(args.device)

        self.layers1 = nn.ModuleList(
            [SBGNNLayer(edgelist_pos, edgelist_neg,
                    dataset_name=dataset_name, emb_size_a=32, emb_size_b=32, aggregator=aggregator1) for _ in range(layer_num)]
        )

    def get_embeddings(self):
        emb_a = self.features_a(torch.arange(self.set_a_num).to(args.device))   #使用节点A的整数索引生成对应的嵌入表示 emb_a。这个过程是为了获取节点A在模型中学到的低维表示，以便在神经网络中进行进一步的计算。
        emb_b = self.features_b(torch.arange(self.set_b_num).to(args.device))
        for m in self.layers1:
            emb1 = m(emb_a, emb_b) # emb = [emb1, emb2]
        return emb1     #这一步就是在生成最终的节点特征

    def multi_view(self, emb1):
        # emb1 = self.get_embeddings()
        emb2 = lil_matrix(emb1.detach().numpy())
        delta = 0.375
        matrix = generate_new_adj(emb2.todense(), delta)
        tensor = torch.tensor(matrix.todense())
        # matrix_b = generate_new_adj(emb1.todense(), delta)
        edge_index = torch.nonzero(tensor, as_tuple=False).t()
        # emb1 = emb1
        # print('debug', emb2.shape[1])

        # edge_index_b = adjacency_matrix_to_edge_index(matrix_b)
        # model_1 = GraphSAGE_NET(input_dim=emb2.shape[1], hidden=64, classes=34)
        # model_b_1 = GraphSAGE_NET(emb1, 16, 2).to(device)

        emb_1 = self.multi_view_model(emb1, edge_index)
        # emb_1 = model_1(emb1, edge_index)
        # emb_b1 = model_b_1(emb1, edge_index_b)
        return emb_1

    def multi_gnn(self, emb, edgelist):
        # edgelist[:, 1] = edgelist[:,1]+515
        edge_index1 = np.vstack([edgelist[:, 0], edgelist[:, 1]])
        edge_index1 = torch.tensor(edge_index1)
        # emb = self.get_embeddings()
        # G = nx.Graph()
        # G.add_edges_from(self.trainedgelist)

        # model1 = GAT_NET(input_dim=emb.shape[1], hidden=16, classes=32, heads=4, num_hops=2)
        # model2 = GAT_NET(input_dim=emb.shape[1], hidden=16, classes=32, heads=4, num_hops=2)
        emb1 = self.multi_gnn_model1(emb, edge_index1).to(device)
        # emb2 = self.multi_gnn_model2(emb, edge_index1).to(device)

        # optimizer_1 = optim.Adam(model1.parameters(), lr=0.5, weight_decay=5e-4)  #0.1
        # optimizer_2 = optim.Adam(model2.parameters(), lr=0.1, weight_decay=5e-4)
        # lr_scheduler_1 = optim.lr_scheduler.StepLR(optimizer_1, step_size=100, gamma=0.1)

        # kl = nn.KLDivLoss(reduction='none').to(device)
        # loss_1 = kl(F.log_softmax(emb1, dim=1), F.softmax(emb2.detach(), dim=1)).sum(dim=1) / emb1.shape[0]
        # loss_1 = torch.sum(loss_1)

        all_nodes = list(set(self.edge_pos.keys()).union(*self.edge_pos.values()))

        # 创建邻接矩阵
        adjacency_matrix1 = np.zeros((len(all_nodes), len(all_nodes)))

        # 填充邻接矩阵
        for start_node, end_nodes in self.edge_pos.items():
            for end_node in end_nodes:
                start_index = all_nodes.index(start_node)
                end_index = all_nodes.index(end_node)
                adjacency_matrix1[start_index, end_index] = 1
        adjacency_matrix1 = torch.tensor(adjacency_matrix1).float().to(device)
        adjacency_matrix1 = torch.matmul(adjacency_matrix1, adjacency_matrix1)
        adjacency_matrix1.fill_diagonal_(0)

        return emb1, adjacency_matrix1

    def forward(self, edgelist):
        emb1 = self.get_embeddings()
        ## embedding1 = self.multi_view(emb1)

        # print("线性层之前embedding1：", embedding1)
        # embedding1 = self.update_func(embedding1)
        # print("线性层之后embedding1：", embedding1)
        # embedding1 = torch.unsqueeze(embedding1, dim=-1)
        # embedding1 = preprocess_features(embedding1)
        # embedding1 = tf.sparse.SparseTensor(indices=embedding1[0], values=embedding1[1], dense_shape=embedding1[2])
        # embedding1 = embedding1.view(-1, -1, -1)
        emb1, adjacency_matrix1 = self.multi_gnn(emb1, edgelist)
        # print("线性层之前emb：", emb)
        # emb = self.update_func(emb)
        # print("线性层之后emb：", emb)
        ## embedding1.requires_grad_(True)
        emb1.requires_grad_(True)
        ## weight1 = torch.randn((embedding1.shape[1], 32))
        ## weight2 = torch.randn((emb1.shape[1], 32))
        # embedding1 = tf.sparse.to_dense(embedding1)
        # emb = tf.sparse.to_dense(emb)
        ## x1 = torch.matmul(embedding1.cpu(), weight1)
        ## x2 = torch.matmul(emb1.cpu(), weight2)
        x2 = emb1

        # x1 = torch.unsqueeze(x1, dim=-1)
        # x2 = torch.unsqueeze(x2, dim=-1)
        ## concatenated_features = torch.cat([x1, x2], dim=1)
        concatenated_features = x2.cpu()

        # concatenated_features = self.update_func(concatenated_features)
        # global_avg_pooled_features = torch.mean(concatenated_features, dim=1, keepdim=True)
        # # 2. 全局平均池化
        # global_avg_pooled_features = torch.mean(concatenated_features, dim=1)
        # classifier = SimpleClassifier(input_size=concatenated_features.size(1), output_size=1)
        # logits = self.classifier(classifier)
        logits = self.classifier(concatenated_features)
        probs = logits.squeeze(dim=1)
        # probs = torch.sigmoid(logits)

        edge_index = edgelist[:, :2]
        # edge_index = tf.convert_to_tensor(edge_index)
        edge_src = edge_index[:, 0]
        edge_dst = edge_index[:, 1]
        src_node_features = torch.zeros(edge_src.size)
        dst_node_features = torch.zeros(edge_dst.size)
        # scalar_value = probs.detach().numpy()
        for i in range(edge_src.shape[0]):
            src_node_features[i] = probs[edge_src[i]]
        for i in range(edge_dst.shape[0]):
            # dst_node_features.append(probs[edge_dst[i]])
            dst_node_features[i] = probs[edge_dst[i]]
        # src_node_features.requires_grad_(True)
        # dst_node_features.requires_grad_(True)
        # edge_features = torch.cat([src_node_features, dst_node_features], dim=-1)
        edge_features = src_node_features + dst_node_features
        # edge_features.requires_grad_(True)

        y = torch.sigmoid(edge_features)

        return y, emb1, adjacency_matrix1


    def loss_multi_gnn_loss(self, emb1, adjacency_matrix1):
        kl = nn.KLDivLoss(reduction='none').to(device)

        total_loss = 0.0
        num_nodes = len(adjacency_matrix1)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = adjacency_matrix1[i][j]
                if weight >= 5:
                    total_loss += weight * similarity_loss(emb1[i], emb1[j])
                else:
                    total_loss += weight * dissimilarity_loss(emb1[i], emb1[j])

        # loss_1 = kl(F.log_softmax(emb1, dim=1), F.softmax(emb2.detach(), dim=1)).sum(dim=1) / emb1.shape[0]

        # max_column_indices = torch.argmax(adjacency_matrix1, dim=1)
        # for i in range(adjacency_matrix1.shape[0]):
        #     loss_1 = kl(F.log_softmax(emb1[i], dim=0), F.softmax(emb1[max_column_indices[i]], dim=0)).sum(
        #                                  dim=0) / adjacency_matrix1.shape[0]
        return total_loss


    def loss_all(self, pred_y, y):
        # result = torch.where(pred_y >= 0.5, torch.tensor(1, dtype=torch.float), torch.tensor(0, dtype=torch.float)).requires_grad_()
        # criterion = FocalLoss()
        pred_y = torch.where(pred_y >= 0.5, 1, 0).float()
        # pred_y[pred_y >= 0.5] = 1  # 预测值大于等于0.5视为预测的符号为1   预测值小于0.5视为预测的符号为0
        # pred_y[pred_y < 0.5] = 0
        assert y.min() >= 0, 'must 0~1'
        assert pred_y.size() == y.size(), 'must be same length'
        criterion = FocalLoss()
        # pos_ratio = y.sum() / y.size()[0]
        # weight = torch.where(y > 0.5, 1. / pos_ratio, 1. / (1 - pos_ratio))
        # weight = torch.where(y > 0.5, (1-pos_ratio), pos_ratio)
        # return F.binary_cross_entropy(result, y, weight=weight)/100
        # loss = criterion(pred_y, y)
        return criterion(pred_y, y)

    # criterion = FocalLoss()
    # loss = criterion(model_output, true_labels)


# =========== function
def load_data(dataset_name):
    train_file_path = os.path.join('experiments-data', f'{dataset_name}_training.txt')
    val_file_path = os.path.join('experiments-data', f'{dataset_name}_validation.txt')
    test_file_path = os.path.join('experiments-data', f'{dataset_name}_testing.txt')

    train_edgelist = []
    with open(train_file_path) as f:
        for ind, line in enumerate(f):
            if ind == 0: continue
            a, b, s = map(int, line.split('\t'))    #对于训练集中的每条边，它由两个节点 a 和 b 组成，并且有一个标签或者权重 s
            train_edgelist.append((a, b, s))

    val_edgelist = []
    with open(val_file_path) as f:
        for ind, line in enumerate(f):
            if ind == 0: continue
            a, b, s = map(int, line.split('\t'))
            val_edgelist.append((a, b, s))

    test_edgelist = []
    with open(test_file_path) as f:
        for ind, line in enumerate(f):
            if ind == 0: continue
            a, b, s = map(int, line.split('\t'))
            test_edgelist.append((a, b, s))

    return np.array(train_edgelist), np.array(val_edgelist), np.array(test_edgelist)


# ============= load data
def load_edgelists(edge_lists):

    edgelist_pos, edgelist_neg = defaultdict(list), defaultdict(list)
    # edgelist_a_b_pos, edgelist_a_b_neg = defaultdict(list), defaultdict(list)  # 创建了两个默认值为列表的字典  将正边和负边分别存储在不同的字典当中
    # edgelist_b_a_pos, edgelist_b_a_neg = defaultdict(list), defaultdict(list)
    #
    # for a, b, s in edge_lists:
    #     if s == 1:          #edgelist_a_b_pos 存储了从节点 a 到节点 b 的正类别的边，edgelist_a_a_neg 存储了从节点 a 到节点 a 的负类别的边
    #         edgelist_a_b_pos[a].append(b)     #a-b正类型的边
    #         edgelist_b_a_pos[b].append(a)     #b-a正类型的边
    #     elif s== -1:
    #         edgelist_a_b_neg[a].append(b)     #a-b负类型的边
    #         edgelist_b_a_neg[b].append(a)     #b-a负类型的边
    #     else:
    #         print(a, b, s)
    #         raise Exception("s must be -1/1")

    for a, b, s in edge_lists:
        if s == 1:          #edgelist_a_b_pos 存储了从节点 a 到节点 b 的正类别的边，edgelist_a_a_neg 存储了从节点 a 到节点 a 的负类别的边
            edgelist_pos[a].append(b)  #正类型的边
            edgelist_pos[b].append(a)
        elif s == -1:
            edgelist_neg[a].append(b)   #负类型的边
            edgelist_neg[b].append(a)
        else:
            print(a, b, s)
            raise Exception("s must be -1/1")      #异常（Exception）抛出语句

    return edgelist_pos, edgelist_neg


@torch.no_grad()
def test_and_val(pred_y, y, mode='val', epoch=0):
    preds = pred_y.cpu().numpy()
    # preds = pred_y.numpy()
    y = y.cpu().numpy()

    preds[preds >= 0.5]  = 1         #预测值大于等于0.5视为预测的符号为1   预测值小于0.5视为预测的符号为0
    preds[preds < 0.5] = 0
    test_y = y

    auc = roc_auc_score(test_y, preds)    #计算二分类模型的性能：在测试集上计算得到的 ROC-AUC 分数
    f1 = f1_score(test_y, preds)
    macro_f1 = f1_score(test_y, preds, average='macro')  #macro_f1 计算的是宏平均 F1 分数，而 micro_f1 计算的是微平均 F1 分数,微平均 F1 将所有类别视为一个整体，考虑了每个样本的贡献。它适用于类别之间样本数量相对平衡的情况。
    micro_f1 = f1_score(test_y, preds, average='micro')  #宏平均 F1 适用于类别之间样本数量差异较大的情况，它对每个类别的性能都平等对待。
    pos_ratio = np.sum(test_y) /  len(test_y)    #这行代码计算了在二分类问题中正类别（positive class）在测试集中的比例
    res = {
        f'{mode}_auc': auc,
        f'{mode}_f1' : f1,
        f'{mode}_pos_ratio': pos_ratio,
        f'{mode}_epoch': epoch,
        f'{mode}_macro_f1' : macro_f1,
        f'{mode}_micro_f1' : micro_f1,
    }
    for k, v in res.items():     #使用 TensorBoard 记录训练过程中的指标信息
        mode ,_, metric = k.partition('_')
        tb_writer.add_scalar(f'{metric}/{mode}', v, epoch)
    # tb_writer.add_scalar( f'{mode}_auc', auc, epoch)
    # tb_writer.add_scalar( f'{mode}_f1', auc, epoch)
    return res



def run():
    train_edgelist, val_edgelist, test_edgelist  = load_data(args.dataset_name)

    set_a_num, set_b_num = DATA_EMB_DIC[args.dataset_name]
    train_edgelist[:, 1] = train_edgelist[:, 1] + set_a_num
    val_edgelist[:, 1] = val_edgelist[:, 1] + set_a_num
    test_edgelist[:, 1] = test_edgelist[:, 1] + set_a_num

    # train_edgelist[:, 1] = train_edgelist[:, 1]
    # val_edgelist[:, 1] = val_edgelist[:, 1]
    # test_edgelist[:, 1] = test_edgelist[:, 1]

    train_y = np.array([i[-1] for i in train_edgelist])   #[i[-1] for i in train_edgelist] 是一个列表推导式，用于从 train_edgelist 中提取每个三元组的最后一个元素（即标签），形成一个新的列表。
    val_y   = np.array([i[-1] for i in val_edgelist])  #np.array(...) 将这个新的列表转换为 NumPy 数组，最终存储在变量 train_y 中
    test_y  = np.array([i[-1] for i in test_edgelist])

    train_y = torch.from_numpy( (train_y + 1)/2 ).float().to(args.device)   #(train_y + 1)/2 对张量中的每个元素进行操作：首先将每个元素加 1，然后再除以 2。这个操作的目的可能是将标签从原先的范围（可能是 -1 和 1）缩放到新的范围（0 和 1）
    val_y = torch.from_numpy( (val_y + 1)/2 ).float().to(args.device)
    test_y = torch.from_numpy( (test_y + 1)/2 ).float().to(args.device)
    # get edge lists
    edgelists = load_edgelists(train_edgelist)

    # model1 = Multi_gnn(edgelists, train_edgelist, dataset_name=args.dataset_name)
    # model1 = model1.to(args.device)
    # optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    model = SBGNN(edgelists, train_edgelist, dataset_name=args.dataset_name, layer_num=args.gnn_layer_num, aggregator1=MeanAggregator)
    model = model.to(args.device)

    print(model.train())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    res_best = {'val_auc': 0}
    for epoch in tqdm(range(1, args.epoch + 2)):   #这行代码使用了 tqdm 函数创建了一个进度条，用于迭代训练过程的每个 epoch.tqdm 是一个 Python 库，用于在命令行中创建美观的进度条

        model.train()
        optimizer.zero_grad()
        # pred_y = model(train_edgelist, pred_y1, emb_1)
        pred_y, emb1, adjacency_matrix1 = model(train_edgelist)
        loss = 0.7 * model.loss_all(pred_y, train_y) + \
               0.3 * model.loss_multi_gnn_loss(emb1, adjacency_matrix1)
        loss = torch.sum(loss)/pred_y.shape[0]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print('loss', loss)


        res_cur = {}
        # if epoch % 5 == 0:
        if True:
        # val/test
            model.eval()
            pred_y, emb1, emb2, adjacency_matrix1 = model(train_edgelist)
            res = test_and_val(pred_y, train_y, mode='train', epoch=epoch)
            res_cur.update(res)   #res_cur 是一个字典，它可能包含了一些先前的评估结果。res 是另一个字典，其中包含了新的评估结果，例如在验证集或测试集上计算得到的AUC等指标。
            pred_val_y, emb1, emb2, adjacency_matrix1 = model(val_edgelist)  #   update 方法用于将字典 res 中的键值对添加到字典 res_cur 中。如果 res 中的键在 res_cur 中不存在，则将该键值对添加到 res_cur 中；如果 res_cur 中已经有相同的键，则用 res 中的值更新 res_cur 中的值。
            res = test_and_val(pred_val_y, val_y, mode='val', epoch=epoch)
            res_cur.update(res)
            pred_test_y, emb1, emb2, adjacency_matrix1 = model(test_edgelist)
            res = test_and_val(pred_test_y, test_y, mode='test', epoch=epoch)
            res_cur.update(res)
            if res_cur['val_auc'] > res_best['val_auc']:
                res_best = res_cur
                print(res_best)
    print('Done! Best Results:')
    print(res_best)
    print_list = ['test_auc', 'test_f1', 'test_macro_f1', 'test_micro_f1']
    for i in print_list:   #遍历 print_list 列表中的指标名称，并打印每个指标的最佳结果
        print(i, res_best[i], end=' ')



def main():
    print(" ".join(sys.argv))   #打印了脚本运行时传递的命令行参数
    # this_fpath = os.path.abspath(__file__)   #将当前脚本文件的路径赋值给变量 this_fpath
    # t = subprocess.run(f'cat {this_fpath}', shell=True, stdout=subprocess.PIPE)   #这行代码的目的是运行 cat 命令，将当前脚本文件的内容输出到标准输出，并将输出结果存储在变量 t 中。((没看懂？？？？))
    # print(str(t.stdout, 'utf-8'))    # 打印了捕获到的脚本文件的内容。
    print('=' * 20)  #打印了一行等号作为分隔线
    run()   #run() 函数的调用表示在执行 main() 函数的过程中会执行 run() 函数的内容。

if __name__ == "__main__":
    main()
