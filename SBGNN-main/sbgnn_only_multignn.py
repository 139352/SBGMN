#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
# from scipy.sparse import lil_matrix

import numpy as np
# import networkx as nx
# from ppo_net import PPO
# from common import get_center_similarity, get_structure_loss, preprocess_features, adjacency_matrix_to_edge_index
from common import weisfeiler_lehman_labels_from_edge_list
# from common import squeeze_tensor_dim_l, edges_to_adjacency_matrix

# import tensorflow as tf
# from keras import backend as K
# from keras.layers import Lambda

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, WLConv
# from torch.distributions import Categorical
# import torch.optim as optim


# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score


from tqdm import tqdm

# import scipy.sparse as sp
# from scipy.spatial.distance import pdist, squareform

import logging

# https://docs.python.org/3/howto/logging.html#logging-advanced-tutorial


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dirpath', default=BASE_DIR, help='Current Dir')
parser.add_argument('--device', type=str, default='cuda:1', help='Devices')
parser.add_argument('--dataset_name', type=str, default='house1to10-1')
parser.add_argument('--a_emb_size', type=int, default=32, help='Embeding A Size')
parser.add_argument('--b_emb_size', type=int, default=32, help='Embeding B Size')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight Decay')
parser.add_argument('--lr', type=float, default=0.005, help='Learning Rate')
parser.add_argument('--seed', type=int, default=13, help='Random seed')
parser.add_argument('--epoch', type=int, default=1000, help='Epoch')
parser.add_argument('--gnn_layer_num', type=int, default=1, help='GNN Layer')
parser.add_argument('--batch_size', type=int, default=500, help='Batch Size')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')#0.5
# parser.add_argument('--agg', type=str, default='AttentionAggregator', choices=['AttentionAggregator', 'MeanAggregator'],
#                     help='Aggregator')
args = parser.parse_args()

# TODO

exclude_hyper_params = ['dirpath', 'device']
hyper_params = dict(vars(args))
for exclude_p in exclude_hyper_params:
    del hyper_params[exclude_p]

hyper_params = "~".join([f"{k}-{v}" for k, v in hyper_params.items()])

from torch.utils.tensorboard import SummaryWriter

# https://pytorch.org/docs/stable/tensorboard.html
tb_writer = SummaryWriter(comment=hyper_params)


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# setup seed
setup_seed(args.seed)

from common import DATA_EMB_DIC

# args.device = 'cpu'
args.device = torch.device(args.device)

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

#定义相同性损失
def similarity_loss(features_i, features_j): # delta stance 正比于 loss
    return 0.5 * torch.norm(features_i - features_j, p=2) ** 2

class AttentionAggregator(nn.Module):
    def __init__(self, a_dim, b_dim):
        super(AttentionAggregator, self).__init__()

        self.out_mlp_layer = nn.Sequential(
            nn.Linear(b_dim, b_dim),
        )
        # self.multi_gnn_model1 = GAT_NET(input_dim=34, hidden=16, classes=34, heads=4, num_hops=2)
        # self.multi_gnn_model2 = GAT_NET(input_dim=34, hidden=16, classes=34, heads=4, num_hops=2)
        # self.gat_model = GATModel(34, 32, 16, 4)

        self.a = nn.Parameter(torch.FloatTensor(a_dim + b_dim, 1))
        nn.init.kaiming_normal_(self.a.data)

    def forward(self, edge_dic_list: dict, feature_a, feature_b, node_num_a, node_num_b):
        edges = []
        for node in range(node_num_a):
            neighs = np.array(edge_dic_list[node]).reshape(-1, 1)
            a = np.array([node]).repeat(len(neighs)).reshape(-1, 1)
            edges.append(np.concatenate([a, neighs], axis=1))

        edges = np.vstack(edges)
        edges = torch.LongTensor(edges).to(args.device)

        new_emb = feature_b
        # emb1 = self.multi_gnn_model1(new_emb, edge_index).to(args.device)
        # emb2 = self.multi_gnn_model2(emb, edge_index).to(args.device)

        new_emb = self.out_mlp_layer(new_emb)
        # output_emb = self.multi_gnn_model1(new_emb, edges.t()).to(args.device)
        # emb2 = self.multi_gnn_model2(new_emb, edges).to(device)


        edge_h_2 = torch.cat([feature_a[edges[:, 0]], new_emb[edges[:, 1]]], dim=1).to(args.device)
        edges_h = torch.exp(F.elu(torch.einsum("ij,jl->il", [edge_h_2, self.a]), 0.1))

        matrix = torch.sparse_coo_tensor(edges.t(), edges_h[:, 0], torch.Size([node_num_a, node_num_b]),
                                         device=args.device)
        # output = self.gat_model(new_emb, matrix)
        row_sum = torch.sparse.mm(matrix, torch.ones(size=(node_num_b, 1)).to(args.device))
        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).to(args.device), row_sum)

        output_emb = torch.sparse.mm(matrix, new_emb)
        output_emb = output_emb.div(row_sum)
        return output_emb


class MeanAggregator(nn.Module):
    def __init__(self, a_dim, b_dim):
        super(MeanAggregator, self).__init__()

        self.out_mlp_layer = nn.Sequential(
            nn.Linear(a_dim, b_dim)
        )

    def forward(self, edge_dic_list: dict, node_num):

        edges = []
        # for node in range(node_num):
        #     neighs = np.array(edge_dic_list[node]).reshape(-1, 1)
        #     a = np.array([node]).repeat(len(neighs)).reshape(-1, 1)
        #     edges.append(np.concatenate([a, neighs], axis=1))
        for node in edge_dic_list:
            neighs = np.array(edge_dic_list[node]).reshape(-1, 1)
            a = np.array([node]).repeat(len(neighs)).reshape(-1, 1)
            edges.append(np.concatenate([a, neighs], axis=1))

        # 计算 WL 标签
        max_iterations = 3  # 设置迭代次数
        output_extra_features = weisfeiler_lehman_labels_from_edge_list(edges, max_iterations, node_num)

        return output_extra_features


class FocalLoss(nn.Module):
    def __init__(self, alpha=5, gamma=2, logits=False, reduce=True):
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

class SBGNNLayer2(nn.Module):
    def __init__(self, edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg, dataset_name=args.dataset_name,
                 emb_size_a=33, emb_size_b=33, aggregator1=MeanAggregator):
        super(SBGNNLayer2, self).__init__()

        self.set_a_num, self.set_b_num = DATA_EMB_DIC[dataset_name]

        # self.feature_a = feature_a
        # self.feature_b = feature_b
        self.edgelist_a_b_pos, self.edgelist_a_b_neg, self.edgelist_b_a_pos, self.edgelist_b_a_neg = \
            edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg

        self.agg_a_from_b_pos = aggregator1(emb_size_b, emb_size_a)
        self.agg_a_from_b_neg = aggregator1(emb_size_b, emb_size_a)
        self.agg_b_from_a_pos = aggregator1(emb_size_a, emb_size_b)
        self.agg_b_from_a_neg = aggregator1(emb_size_a, emb_size_b)

        self.update_func = nn.Sequential(        #self.update_func 将会是一个由以下几个层组成的神经网络模型。这个模型可能被用于在图神经网络中进行节点特征的更新或聚合操作
            nn.Dropout(args.dropout),
            nn.Linear(emb_size_a + 2, emb_size_a * 4),
            nn.PReLU(),
            nn.Linear(emb_size_b * 4, emb_size_b + 2)
        )

    def forward(self, feature_a, feature_b):
        # assert feature_a.size()[0] == self.set_a_num, 'set_b_num error'
        # assert feature_b.size()[0] == self.set_b_num, 'set_b_num error'
        # feature_a = feature_all[:self.set_a_num]
        # feature_b = feature_all[self.set_a_num:]

        node_num_a, node_num_b = self.set_a_num, self.set_b_num
        # node_num = node_num_a + node_num_b

        self.edgelist_a_b_pos = {key: [value_element + node_num_a for value_element in value_list] for key, value_list in self.edgelist_a_b_pos.items()}
        self.edgelist_a_b_neg = {key: [value_element + node_num_a for value_element in value_list] for key, value_list in self.edgelist_a_b_neg.items()}
        m_a_b_pos = self.agg_a_from_b_pos(self.edgelist_a_b_pos, node_num_a)
        sorted_keys_a_b_pos = sorted(m_a_b_pos.keys())
        tensor_a_b_pos = [float(m_a_b_pos[key]) if key in sorted_keys_a_b_pos else .0 for key in
                          range(1, node_num_a + 1)]
        tensor_a_b_pos = torch.tensor((tensor_a_b_pos), dtype=torch.float).view(-1, 1).to(args.device)
        # tensor_a_b_pos = torch.tensor([float(m_a_b_pos[key]) for key in sorted_keys_a_b_pos
        #    if key in list(range(1, node_num_a + 1))], dtype=torch.float).view(-1, 1).to(args.device)
        # tensor_a_b_pos = tensor_a_b_pos[:node_num_a,:].to(args.device)
        tensor_a_b_pos = F.normalize(tensor_a_b_pos, p=2, dim=0)

        m_a_b_neg = self.agg_a_from_b_neg(self.edgelist_a_b_neg, node_num_a)
        sorted_keys_a_b_neg = sorted(m_a_b_neg.keys())
        tensor_a_b_neg = [float(m_a_b_neg[key]) if key in sorted_keys_a_b_neg else .0 for key in
                          range(1, node_num_a + 1)]
        tensor_a_b_neg = torch.tensor((tensor_a_b_neg), dtype=torch.float).view(-1, 1).to(args.device)
        # tensor_a_b_neg = torch.tensor([float(m_a_b_neg[key]) for key in sorted_keys_a_b_neg
        #    if key in list(range(1, node_num_a + 1))], dtype=torch.float).view(-1, 1).to(args.device)
        # tensor_a_b_neg = tensor_a_b_neg[:node_num_a,:].to(args.device)
        tensor_a_b_neg = F.normalize(tensor_a_b_neg, p=2, dim=0)

        # self.edgelist_b_a_pos = {key: [value_element - node_num_a for value_element in value_list] for key, value_list in self.edgelist_b_a_pos.items()}
        # self.edgelist_b_a_neg = {key: [value_element - node_num_a for value_element in value_list] for key, value_list in self.edgelist_b_a_neg.items()}
        self.edgelist_b_a_pos = {key: [value_element + node_num_b for value_element in value_list] for key, value_list
                                 in self.edgelist_b_a_pos.items()}
        self.edgelist_b_a_neg = {key: [value_element + node_num_b for value_element in value_list] for key, value_list
                                 in self.edgelist_b_a_neg.items()}

        m_b_a_pos = self.agg_b_from_a_pos(self.edgelist_b_a_pos, node_num_b)
        sorted_keys_b_a_pos = sorted(m_b_a_pos.keys())
        tensor_b_a_pos = [float(m_b_a_pos[key]) if key in sorted_keys_b_a_pos else .0 for key in
                          range(1, node_num_b + 1)]
        tensor_b_a_pos = torch.tensor((tensor_b_a_pos), dtype=torch.float).view(-1, 1).to(args.device)
        # tensor_b_a_pos = torch.tensor([float(m_b_a_pos[key]) for key in sorted_keys_b_a_pos
        #    if key in list(range(1, node_num_b + 1))], dtype=torch.float).view(-1, 1).to(args.device)
        # tensor_b_a_pos = tensor_b_a_pos[:node_num_b, :].to(args.device)
        tensor_b_a_pos = F.normalize(tensor_b_a_pos, p=2, dim=0)

        m_b_a_neg = self.agg_b_from_a_neg(self.edgelist_b_a_neg, node_num_b)
        sorted_keys_b_a_neg = sorted(m_b_a_neg.keys())
        tensor_b_a_neg = [float(m_b_a_neg[key]) if key in sorted_keys_b_a_neg else .0 for key in
                          range(1, node_num_b + 1)]
        tensor_b_a_neg = torch.tensor((tensor_b_a_neg), dtype=torch.float).view(-1, 1).to(args.device)
        # tensor_b_a_neg = torch.tensor([float(m_b_a_neg[key]) for key in sorted_keys_b_a_neg
        #    if key in list(range(1, node_num_b + 1))], dtype=torch.float).view(-1, 1).to(args.device)
        # tensor_b_a_neg = tensor_b_a_neg[node_num_a:,:].to(args.device)
        tensor_b_a_neg = F.normalize(tensor_b_a_neg, p=2, dim=0)

        new_feature_a = torch.cat([feature_a, tensor_a_b_pos, tensor_a_b_neg], dim=1).to(args.device)
        # new_feature_b = torch.cat([feature_b, tensor_neg], dim=1)
        new_feature_b = torch.cat([feature_b, tensor_b_a_pos, tensor_b_a_neg], dim=1).to(args.device)

        new_feature_a = self.update_func(new_feature_a)
        new_feature_b = self.update_func(new_feature_b)

        return new_feature_a, new_feature_b


class SBGNNLayer(nn.Module):
    def __init__(self, edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,
                 dataset_name=args.dataset_name, emb_size_a=35, emb_size_b=35, aggregator=AttentionAggregator):
        super(SBGNNLayer, self).__init__()
        #
        self.set_a_num, self.set_b_num = DATA_EMB_DIC[dataset_name]

        self.edgelist_a_b_pos, self.edgelist_a_b_neg, self.edgelist_b_a_pos, self.edgelist_b_a_neg = \
            edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg

        self.agg_a_from_b_pos = aggregator(emb_size_b, emb_size_a)
        self.agg_a_from_b_neg = aggregator(emb_size_b, emb_size_a)
        self.agg_b_from_a_pos = aggregator(emb_size_a, emb_size_b)
        self.agg_b_from_a_neg = aggregator(emb_size_a, emb_size_b)


        self.update_func = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(emb_size_a * 3, emb_size_a * 2),
            nn.PReLU(),
            nn.Linear(emb_size_b * 2, emb_size_b)

        )

    def forward(self, feature_a, feature_b):

        node_num_a, node_num_b = self.set_a_num, self.set_b_num

        m_a_from_b_pos = self.agg_a_from_b_pos(self.edgelist_a_b_pos, feature_a, feature_b, node_num_a, node_num_b)
        m_a_from_b_neg = self.agg_a_from_b_neg(self.edgelist_a_b_neg, feature_a, feature_b, node_num_a, node_num_b)

        new_feature_a = torch.cat([feature_a, m_a_from_b_pos, m_a_from_b_neg], dim=1).to(args.device)
        new_feature_a = self.update_func(new_feature_a)

        m_b_from_a_pos = self.agg_b_from_a_pos(self.edgelist_b_a_pos, feature_b, feature_a, node_num_b, node_num_a)
        m_b_from_a_neg = self.agg_b_from_a_neg(self.edgelist_b_a_neg, feature_b, feature_a, node_num_b, node_num_a)

        new_feature_b = torch.cat([feature_b, m_b_from_a_pos, m_b_from_a_neg], dim=1).to(args.device)
        new_feature_b = self.update_func(new_feature_b)

        return new_feature_a, new_feature_b


class SBGNN(nn.Module):
    def __init__(self, edgelists,
                 dataset_name=args.dataset_name, layer_num=1, emb_size_a=32, emb_size_b=32):
        super(SBGNN, self).__init__()

        assert len(edgelists) == 6, 'must 6 edgelists'
        edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg, \
        edgelist_pos, edgelist_neg = edgelists

        self.edgelist_pos = edgelist_pos

        self.set_a_num, self.set_b_num = DATA_EMB_DIC[dataset_name]

        self.features_a = nn.Embedding(self.set_a_num, emb_size_a)
        self.features_b = nn.Embedding(self.set_b_num, emb_size_b)
        # category = torch.full((self.set_a_num, 1), 1, dtype=torch.long).to(args.device)
        # category_expanded = category.expand(self.features_a.size(0), 1)
        # emb_a = torch.cat([self.features_a, category_expanded.float()], dim=1)
        #
        # category2 = torch.full((self.set_b_num, 1), 0, dtype=torch.long).to(args.device)
        # category_expanded2 = category2.expand(self.features_b.size(0), 1)
        # emb_b = torch.cat([self.features_b, category_expanded2.float()], dim=1)

        self.features_a.weight.requires_grad = True
        self.features_b.weight.requires_grad = True
        # self.multi_view_model = GraphSAGE_NET(input_dim=35, hidden=70, classes=35)

        self.layers_gnn1 = nn.ModuleList(
            [SBGNNLayer(edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,
                        dataset_name=dataset_name, emb_size_a=35, emb_size_b=35, aggregator=AttentionAggregator) for _ in
             range(layer_num)]
        )
        self.layers_gnn2 = nn.ModuleList(
            [SBGNNLayer(edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,
                        dataset_name=dataset_name, emb_size_a=35, emb_size_b=35, aggregator=AttentionAggregator) for _ in
             range(layer_num)]
        )

        self.layers2 = nn.ModuleList(
            [SBGNNLayer2(edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,
                        dataset_name=dataset_name, emb_size_a=33, emb_size_b=33, aggregator1=MeanAggregator) for _ in
             range(layer_num)]
        )

    def get_embeddings(self):
        emb_a = self.features_a(torch.arange(self.set_a_num).to(args.device))
        emb_b = self.features_b(torch.arange(self.set_b_num).to(args.device))
        label_a = torch.zeros(self.set_a_num, 1).to(args.device)
        label_b = torch.ones(self.set_b_num, 1).to(args.device)

        emb_a = torch.cat([emb_a, label_a],dim=1)
        emb_b = torch.cat([emb_b, label_b], dim=1)

        for m in self.layers2:
            emb_a, emb_b = m(emb_a, emb_b)
        return emb_a, emb_b

    def gnn1(self, embedding_a, embedding_b):
        for m in self.layers_gnn1:
            embedding_a, embedding_b = m(embedding_a, embedding_b)
        return embedding_a, embedding_b

    def gnn2(self, embedding_a, embedding_b):
        for m in self.layers_gnn2:
            embedding_a, embedding_b = m(embedding_a, embedding_b)
        return embedding_a, embedding_b


    # def multi_view(self, emb_all):
    #     if emb_all.device != 'cpu':
    #         emb_view = lil_matrix(emb_all.cpu().detach().numpy())
    #     else:
    #         emb_view = lil_matrix(emb_all.detach().numpy())
    #     delta = 0.375
    #     matrix = generate_new_adj(emb_view.todense(), delta)
    #     tensor = torch.tensor(matrix.todense())
    #     edge_index = torch.nonzero(tensor, as_tuple=False).t().to(args.device)
    #     emb_view = self.multi_view_model(emb_all, edge_index)
    #     return emb_view


    def forward(self, edge_lists):
        embedding_a, embedding_b = self.get_embeddings()   #特征增强------WL算法
        emb_all = torch.cat([embedding_a, embedding_b], dim=0).to(args.device)
        # emb_view = self.multi_view(emb_all)    #同类型节点特征构造------multi-view
        # emb_view_a = emb_view[:self.set_a_num]
        # emb_view_b = emb_view[self.set_a_num:]
        embedding_a1, embedding_b1 = self.gnn1(embedding_a, embedding_b)
        embedding_a2, embedding_b2 = self.gnn2(embedding_a, embedding_b)

        # embedding_a = torch.cat([embedding_a1, emb_view_a], dim=1).to(args.device)
        # embedding_b = torch.cat([embedding_b1, emb_view_b], dim=1).to(args.device)
        embedding_a = embedding_a1
        embedding_b = embedding_b1

        y = torch.einsum("ij, ij->i", [embedding_a[edge_lists[:, 0]], embedding_b[edge_lists[:, 1]]])
        y = torch.sigmoid(y)

        # y: 0 ~ 1
        # embedding_a1: -1.2 ~ 1
        # embedding_b1: -1.18 ~ 1.3
        # embedding_a2: -1.2 ~ 1.01
        # embedding_b2: -1.18 ~ 1.14
        return y, embedding_a1, embedding_b1, embedding_a2, embedding_b2


    # def view_loss(self, emb_a, emb_b):
    #     label_a = torch.full_like(emb_a, -1)
    #     label_b = torch.ones_like(emb_b)
    #     # label_b = torch.full(emb_b, torch.tensor(1))
    #     loss_a = F.cross_entropy(emb_a, label_a)/emb_a.shape[0]
    #     loss = F.cross_entropy(emb_b, label_b)/emb_b.shape[0] - loss_a
    #     return loss

    def multi_gnn_loss(self, embedding_a, embedding_b, embedding_a2, embedding_b2):
        emb1 = torch.cat([embedding_a, embedding_b], dim=0).to(args.device)
        emb2 = torch.cat([embedding_a2, embedding_b2], dim=0).to(args.device)

        all_nodes = list(set(self.edgelist_pos.keys()).union(*self.edgelist_pos.values()))
        # 创建邻接矩阵
        adjacency_matrix1 = np.zeros((len(all_nodes), len(all_nodes)))
        # 填充邻接矩阵
        for start_node, end_nodes in self.edgelist_pos.items():
            for end_node in end_nodes:
                start_index = all_nodes.index(start_node)
                end_index = all_nodes.index(end_node)
                adjacency_matrix1[start_index, end_index] = 1
        adjacency_matrix1 = torch.tensor(adjacency_matrix1).float().to(args.device)
        adjacency_matrix1 = torch.matmul(adjacency_matrix1, adjacency_matrix1)
        adjacency_matrix1.fill_diagonal_(0)

        adjacency_matrix1[:self.set_a_num, self.set_a_num:] = 0
        adjacency_matrix1[self.set_a_num:, :self.set_a_num] = 0

        loss_1 = F.kl_div(F.log_softmax(emb1, dim=1), F.softmax(emb2, dim=1))

        top_values, top_indices = torch.topk(adjacency_matrix1, k=3, dim=0)

        # 计算相似度损失
        adjacency_loss = 0.0
        for i in range(top_values.shape[0]):  # 遍历每一列
            for j in range(top_values.shape[1]):  # 遍历前十个最大值的索引
                # weight = torch.log(1 + top_values[i, j])
                weight = top_values[i, j]
                adjacency_loss += 1 / (weight + 1) * similarity_loss(emb1[j], emb1[top_indices[i, j]])

        adjacency_loss = adjacency_loss / (top_values.shape[0] * top_values.shape[1]) # to mean

        loss = loss_1 + adjacency_loss
        # loss = loss_1

        return loss


    def loss(self, pred_y, y):
        assert y.min() >= 0, 'must 0~1'
        assert pred_y.size() == y.size(), 'must be same length'
        pos_ratio = y.sum() / y.size()[0]
        weight = torch.where(y > 0.5, 1. / pos_ratio, 1. / (1 - pos_ratio))
        # weight = torch.where(y > 0.5, (1-pos_ratio), pos_ratio)
        # criterion = FocalLoss()
        return F.binary_cross_entropy(pred_y, y, weight=weight)



# =========== function
def load_data(dataset_name):
    train_file_path = os.path.join('experiments-data', f'{dataset_name}_training.txt')
    val_file_path = os.path.join('experiments-data', f'{dataset_name}_validation.txt')
    test_file_path = os.path.join('experiments-data', f'{dataset_name}_testing.txt')


    train_edgelist = []
    with open(train_file_path) as f:
        for ind, line in enumerate(f):
            if ind == 0: continue
            a, b, s = map(int, line.split('\t'))
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
    edgelist_a_b_pos, edgelist_a_b_neg = defaultdict(list), defaultdict(list)
    edgelist_b_a_pos, edgelist_b_a_neg = defaultdict(list), defaultdict(list)
    edgelist_pos, edgelist_neg = defaultdict(list), defaultdict(list)
    edgelist_a_a_pos, edgelist_a_a_neg = defaultdict(list), defaultdict(list)
    edgelist_b_b_pos, edgelist_b_b_neg = defaultdict(list), defaultdict(list)
    # edgelist_a_b_pos2, edgelist_a_b_neg2 = defaultdict(list), defaultdict(list)
    # edgelist_b_a_pos2, edgelist_b_a_neg2 = defaultdict(list), defaultdict(list)

    set_a_num, set_b_num = DATA_EMB_DIC[args.dataset_name]
    for a, b, s in edge_lists:
        if s == 1:
            edgelist_a_b_pos[a].append(b)
            edgelist_b_a_pos[b].append(a)
            edgelist_pos[a].append(b+set_a_num)
            edgelist_pos[b+set_a_num].append(a)
            # edgelist_a_b_pos2[a].append(b+515)
            # edgelist_b_a_pos2[b+515].append(a)
        elif s == -1:
            edgelist_a_b_neg[a].append(b)
            edgelist_b_a_neg[b].append(a)
            edgelist_neg[a].append(b+set_a_num)
            edgelist_neg[b+set_a_num].append(a)
            # edgelist_a_b_neg2[a].append(b+515)
            # edgelist_b_a_neg2[b+515].append(a)
        else:
            print(a, b, s)
            raise Exception("s must be -1/1")

    edge_list_a_a = defaultdict(lambda: defaultdict(int))
    edge_list_b_b = defaultdict(lambda: defaultdict(int))
    for a, b, s in edge_lists:
        for b2 in edgelist_a_b_pos[a]:
            edge_list_b_b[b][b2] += 1 * s
        for b2 in edgelist_a_b_neg[a]:
            edge_list_b_b[b][b2] -= 1 * s
        for a2 in edgelist_b_a_pos[b]:
            edge_list_a_a[a][a2] += 1 * s
        for a2 in edgelist_b_a_neg[b]:
            edge_list_a_a[a][a2] -= 1 * s

    for a1 in edge_list_a_a:
        for a2 in edge_list_a_a[a1]:
            v = edge_list_a_a[a1][a2]
            if a1 == a2: continue
            if v > 0:
                edgelist_a_a_pos[a1].append(a2)
            elif v < 0:
                edgelist_a_a_neg[a1].append(a2)

    for b1 in edge_list_b_b:
        for b2 in edge_list_b_b[b1]:
            v = edge_list_b_b[b1][b2]
            if b1 == b2: continue
            if v > 0:
                edgelist_b_b_pos[b1].append(b2)
            elif v < 0:
                edgelist_b_b_neg[b1].append(b2)

    return edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg, \
           edgelist_pos, edgelist_neg


@torch.no_grad()
def test_and_val(pred_y, y, mode='val', epoch=0):
    preds = pred_y.cpu().numpy()
    y = y.cpu().numpy()

    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    test_y = y

    auc = roc_auc_score(test_y, preds)
    f1 = f1_score(test_y, preds)
    macro_f1 = f1_score(test_y, preds, average='macro')
    micro_f1 = f1_score(test_y, preds, average='micro')
    pos_ratio = np.sum(test_y) / len(test_y)
    res = {
        f'{mode}_auc': auc,
        f'{mode}_f1': f1,
        f'{mode}_pos_ratio': pos_ratio,
        f'{mode}_epoch': epoch,
        f'{mode}_macro_f1': macro_f1,
        f'{mode}_micro_f1': micro_f1,
    }
    for k, v in res.items():
        mode, _, metric = k.partition('_')
        tb_writer.add_scalar(f'{metric}/{mode}', v, epoch)
    # tb_writer.add_scalar( f'{mode}_auc', auc, epoch)
    # tb_writer.add_scalar( f'{mode}_f1', auc, epoch)
    return res


def run():
    train_edgelist, val_edgelist, test_edgelist = load_data(args.dataset_name)

    set_a_num, set_b_num = DATA_EMB_DIC[args.dataset_name]
    train_y = np.array([i[-1] for i in train_edgelist])
    val_y = np.array([i[-1] for i in val_edgelist])
    test_y = np.array([i[-1] for i in test_edgelist])

    train_y = torch.from_numpy((train_y + 1) / 2).float().to(args.device)
    val_y = torch.from_numpy((val_y + 1) / 2).float().to(args.device)
    test_y = torch.from_numpy((test_y + 1) / 2).float().to(args.device)
    # get edge lists
    edgelists = load_edgelists(train_edgelist)

    # if args.agg == 'MeanAggregator':
    #     agg = MeanAggregator
    # else:
    #     agg = AttentionAggregator

    model = SBGNN(edgelists, dataset_name=args.dataset_name, layer_num=args.gnn_layer_num)
    model = model.to(args.device)

    # print(model.train())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-8)

    res_best = {'test_auc': 0}
    for epoch in tqdm(range(1, args.epoch + 2)):
        # train
        model.train()
        optimizer.zero_grad()
        pred_y, embedding_a, embedding_b, embedding_a2, embedding_b2 = model(train_edgelist)
        # print('debug', pred_y)
        loss1 = model.loss(pred_y, train_y)
        # loss2 = model.view_loss(emb_view_a, emb_view_b)
        loss3 = model.multi_gnn_loss(embedding_a, embedding_b, embedding_a2, embedding_b2)
        # loss = 0.8*loss1 + 0.15*loss2 + 0.05*loss3
        loss = 0.8 * loss1 + 0.2 * loss3
        # loss = loss1
        loss.backward()
        optimizer.step()
        print('loss', loss, 'lr', scheduler.get_last_lr())
        scheduler.step()

        res_cur = {}
        # if epoch % 5 == 0:
        if True:
            # val/test
            model.eval()
            pred_y, embedding_a, embedding_b, embedding_a2, embedding_b2 = model(train_edgelist)
            res = test_and_val(pred_y, train_y, mode='train', epoch=epoch)
            res_cur.update(res)
            pred_val_y, embedding_a, embedding_b, embedding_a2, embedding_b2 = model(val_edgelist)
            res = test_and_val(pred_val_y, val_y, mode='val', epoch=epoch)
            res_cur.update(res)
            pred_test_y, embedding_a, embedding_b, embedding_a2, embedding_b2  = model(test_edgelist)
            res = test_and_val(pred_test_y, test_y, mode='test', epoch=epoch)
            res_cur.update(res)
            if res_cur['test_auc'] > res_best['test_auc']:
                res_best = res_cur
                # print(res_best)
            print(res_cur)
    print('Done! Best Results:')
    print(res_best)
    print_list = ['test_auc', 'test_f1', 'test_macro_f1', 'test_micro_f1']
    for i in print_list:
        print(i, res_best[i], end=' ')


def main():
    print(" ".join(sys.argv))
    this_fpath = os.path.abspath(__file__)
    t = subprocess.run(f'cat {this_fpath}', shell=True, stdout=subprocess.PIPE)
    print(str(t.stdout, 'utf-8'))
    print('=' * 20)
    run()


if __name__ == "__main__":
    main()
