from scipy.sparse import lil_matrix
import numpy as np

from common import weisfeiler_lehman_labels_from_edge_list, generate_new_adj
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from common import DATA_EMB_DIC

#定义相同性损失
def similarity_loss(features_i, features_j): # delta stance 正比于 loss
    return 0.5 * torch.norm(features_i - features_j, p=2) ** 2

class AttentionAggregator(nn.Module):
    def __init__(self, a_dim, b_dim):
        super(AttentionAggregator, self).__init__()

        self.out_mlp_layer = nn.Sequential(
            nn.Linear(b_dim, b_dim),
        )

        self.a = nn.Parameter(torch.FloatTensor(a_dim + b_dim, 1))
        nn.init.kaiming_normal_(self.a.data)

    def forward(self, edge_dic_list: dict, feature_a, feature_b, node_num_a, node_num_b):
        device = feature_a.device

        edges = []
        for node in range(node_num_a):
            neighs = np.array(edge_dic_list[node]).reshape(-1, 1)
            a = np.array([node]).repeat(len(neighs)).reshape(-1, 1)
            edges.append(np.concatenate([a, neighs], axis=1))

        edges = np.vstack(edges)
        edges = torch.LongTensor(edges).to(device)

        new_emb = feature_b
        new_emb = self.out_mlp_layer(new_emb)

        edge_h_2 = torch.cat([feature_a[edges[:, 0]], new_emb[edges[:, 1]]], dim=1).to(device)
        edges_h = torch.exp(F.elu(torch.einsum("ij,jl->il", [edge_h_2, self.a]), 0.1))

        matrix = torch.sparse_coo_tensor(edges.t(), edges_h[:, 0], torch.Size([node_num_a, node_num_b]),
                                         device=device)
        # output = self.gat_model(new_emb, matrix)
        row_sum = torch.sparse.mm(matrix, torch.ones(size=(node_num_b, 1)).to(device))
        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).to(device), row_sum)

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

    def __init__(self, input_dim, hidden, output_dim, dropout):
        super(GraphSAGE_NET, self).__init__()
        self.dropout=dropout
        self.sage1 = SAGEConv(input_dim, hidden)  # 定义两层GraphSAGE层
        self.sage2 = SAGEConv(hidden, output_dim)

    def forward(self, x, edge_index):

        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.sage2(x, edge_index)

        return x




class SBGMNLayer2(nn.Module):
    def __init__(self, edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg, dataset_name, dropout,
                 emb_size_a, emb_size_b, aggregator1=MeanAggregator):
        super(SBGMNLayer2, self).__init__()
        self.set_a_num, self.set_b_num = DATA_EMB_DIC[dataset_name]
        self.dropout = dropout
        self.edgelist_a_b_pos, self.edgelist_a_b_neg, self.edgelist_b_a_pos, self.edgelist_b_a_neg = \
            edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg

        self.agg_a_from_b_pos = aggregator1(emb_size_b, emb_size_a)
        self.agg_a_from_b_neg = aggregator1(emb_size_b, emb_size_a)
        self.agg_b_from_a_pos = aggregator1(emb_size_a, emb_size_b)
        self.agg_b_from_a_neg = aggregator1(emb_size_a, emb_size_b)

        self.update_func = nn.Sequential(        #self.update_func 将会是一个由以下几个层组成的神经网络模型。这个模型可能被用于在图神经网络中进行节点特征的更新或聚合操作
            nn.Dropout(self.dropout),
            nn.Linear(emb_size_a + 2, emb_size_a * 4),
            nn.PReLU(),
            nn.Linear(emb_size_b * 4, emb_size_b + 2)
        )

    def forward(self, feature_a, feature_b):

        device = feature_a.device

        node_num_a, node_num_b = self.set_a_num, self.set_b_num
        # node_num = node_num_a + node_num_b

        self.edgelist_a_b_pos = {key: [value_element + node_num_a for value_element in value_list] for key, value_list in self.edgelist_a_b_pos.items()}
        self.edgelist_a_b_neg = {key: [value_element + node_num_a for value_element in value_list] for key, value_list in self.edgelist_a_b_neg.items()}
        m_a_b_pos = self.agg_a_from_b_pos(self.edgelist_a_b_pos, node_num_a)
        sorted_keys_a_b_pos = sorted(m_a_b_pos.keys())
        tensor_a_b_pos = [float(m_a_b_pos[key]) if key in sorted_keys_a_b_pos else .0 for key in
                          range(1, node_num_a + 1)]
        tensor_a_b_pos = torch.tensor((tensor_a_b_pos), dtype=torch.float).view(-1, 1).to(device)
        # tensor_a_b_pos = torch.tensor([float(m_a_b_pos[key]) for key in sorted_keys_a_b_pos
        #    if key in list(range(1, node_num_a + 1))], dtype=torch.float).view(-1, 1).to(args.device)
        # tensor_a_b_pos = tensor_a_b_pos[:node_num_a,:].to(args.device)
        tensor_a_b_pos = F.normalize(tensor_a_b_pos, p=2, dim=0)

        m_a_b_neg = self.agg_a_from_b_neg(self.edgelist_a_b_neg, node_num_a)
        sorted_keys_a_b_neg = sorted(m_a_b_neg.keys())
        tensor_a_b_neg = [float(m_a_b_neg[key]) if key in sorted_keys_a_b_neg else .0 for key in
                          range(1, node_num_a + 1)]
        tensor_a_b_neg = torch.tensor((tensor_a_b_neg), dtype=torch.float).view(-1, 1).to(device)

        tensor_a_b_neg = F.normalize(tensor_a_b_neg, p=2, dim=0)

        self.edgelist_b_a_pos = {key: [value_element + node_num_b for value_element in value_list] for key, value_list
                                 in self.edgelist_b_a_pos.items()}
        self.edgelist_b_a_neg = {key: [value_element + node_num_b for value_element in value_list] for key, value_list
                                 in self.edgelist_b_a_neg.items()}

        m_b_a_pos = self.agg_b_from_a_pos(self.edgelist_b_a_pos, node_num_b)
        sorted_keys_b_a_pos = sorted(m_b_a_pos.keys())
        tensor_b_a_pos = [float(m_b_a_pos[key]) if key in sorted_keys_b_a_pos else .0 for key in
                          range(1, node_num_b + 1)]
        tensor_b_a_pos = torch.tensor((tensor_b_a_pos), dtype=torch.float).view(-1, 1).to(device)

        tensor_b_a_pos = F.normalize(tensor_b_a_pos, p=2, dim=0)

        m_b_a_neg = self.agg_b_from_a_neg(self.edgelist_b_a_neg, node_num_b)
        sorted_keys_b_a_neg = sorted(m_b_a_neg.keys())
        tensor_b_a_neg = [float(m_b_a_neg[key]) if key in sorted_keys_b_a_neg else .0 for key in
                          range(1, node_num_b + 1)]
        tensor_b_a_neg = torch.tensor((tensor_b_a_neg), dtype=torch.float).view(-1, 1).to(device)

        tensor_b_a_neg = F.normalize(tensor_b_a_neg, p=2, dim=0)

        new_feature_a = torch.cat([feature_a, tensor_a_b_pos, tensor_a_b_neg], dim=1).to(device)
        # new_feature_b = torch.cat([feature_b, tensor_neg], dim=1)
        new_feature_b = torch.cat([feature_b, tensor_b_a_pos, tensor_b_a_neg], dim=1).to(device)

        new_feature_a = self.update_func(new_feature_a)
        new_feature_b = self.update_func(new_feature_b)

        return new_feature_a, new_feature_b



class SBGMNLayer(nn.Module):
    def __init__(self, edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,
                 dataset_name, dropout, emb_size_a, emb_size_b, aggregator=AttentionAggregator):
        super(SBGMNLayer, self).__init__()
        self.dropout = dropout
        self.set_a_num, self.set_b_num = DATA_EMB_DIC[dataset_name]

        self.edgelist_a_b_pos, self.edgelist_a_b_neg, self.edgelist_b_a_pos, self.edgelist_b_a_neg = \
            edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg

        self.agg_a_from_b_pos = aggregator(emb_size_b, emb_size_a)
        self.agg_a_from_b_neg = aggregator(emb_size_b, emb_size_a)
        self.agg_b_from_a_pos = aggregator(emb_size_a, emb_size_b)
        self.agg_b_from_a_neg = aggregator(emb_size_a, emb_size_b)

        self.update_func = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(emb_size_a * 3, emb_size_a * 2),
            nn.PReLU(),
            nn.Linear(emb_size_b * 2, emb_size_b)

        )

    def forward(self, feature_a, feature_b):
        device = feature_a.device

        node_num_a, node_num_b = self.set_a_num, self.set_b_num

        m_a_from_b_pos = self.agg_a_from_b_pos(self.edgelist_a_b_pos, feature_a, feature_b, node_num_a, node_num_b)
        m_a_from_b_neg = self.agg_a_from_b_neg(self.edgelist_a_b_neg, feature_a, feature_b, node_num_a, node_num_b)

        new_feature_a = torch.cat([feature_a, m_a_from_b_pos, m_a_from_b_neg], dim=1).to(device)
        new_feature_a = self.update_func(new_feature_a)

        m_b_from_a_pos = self.agg_b_from_a_pos(self.edgelist_b_a_pos, feature_b, feature_a, node_num_b, node_num_a)
        m_b_from_a_neg = self.agg_b_from_a_neg(self.edgelist_b_a_neg, feature_b, feature_a, node_num_b, node_num_a)

        new_feature_b = torch.cat([feature_b, m_b_from_a_pos, m_b_from_a_neg], dim=1).to(device)
        new_feature_b = self.update_func(new_feature_b)

        return new_feature_a, new_feature_b


class SBGMN(nn.Module):
    def __init__(self, edgelists, dataset_name, layer_num, view_hidden, emb_size_a, emb_size_b, dropout=0.5,
                 view_relate_rate = 0.05, device='cuda'):
        super(SBGMN, self).__init__()

        assert len(edgelists) == 6, 'must 6 edgelists'
        edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg, \
        edgelist_pos, edgelist_neg = edgelists

        self.edgelist_pos = edgelist_pos

        self.set_a_num, self.set_b_num = DATA_EMB_DIC[dataset_name]

        self.device = device
        self.view_relate_rate = view_relate_rate

        self.features_a = nn.Embedding(self.set_a_num, emb_size_a)
        self.features_b = nn.Embedding(self.set_b_num, emb_size_b)

        # self.features_a.weight.requires_grad = True
        # self.features_b.weight.requires_grad = True
        self.multi_left_view_model = GraphSAGE_NET(input_dim=emb_size_a+3, hidden=view_hidden, output_dim=emb_size_a+3, dropout=dropout)  # 1316
        self.multi_right_view_model = GraphSAGE_NET(input_dim=emb_size_a+3, hidden=view_hidden, output_dim=emb_size_a+3, dropout=dropout)  # 1316

        self.layers_gnn1 = nn.ModuleList(
            [SBGMNLayer(edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,
                dataset_name=dataset_name, dropout=dropout, emb_size_a=emb_size_a+3, emb_size_b=emb_size_b+3, aggregator=AttentionAggregator) for _ in
             range(layer_num)]
        )
        self.layers_gnn2 = nn.ModuleList(
            [SBGMNLayer(edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,
                dataset_name=dataset_name, dropout=dropout, emb_size_a=emb_size_a+3, emb_size_b=emb_size_b+3, aggregator=AttentionAggregator) for _ in
             range(layer_num)]
        )

        self.layers2 = nn.ModuleList(
            [SBGMNLayer2(edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,
                dataset_name=dataset_name, dropout=dropout, emb_size_a=emb_size_a+1, emb_size_b=emb_size_b+1, aggregator1=MeanAggregator) for _ in
             range(1)]
        )

    def get_embeddings(self):
        emb_a = self.features_a(torch.arange(self.set_a_num).to(self.device))
        emb_b = self.features_b(torch.arange(self.set_b_num).to(self.device))
        label_a = torch.zeros(self.set_a_num, 1).to(self.device)
        label_b = torch.ones(self.set_b_num, 1).to(self.device)

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

    def multi_view(self, emb_all):

        if emb_all.device != 'cpu':
            emb_view = lil_matrix(emb_all.cpu().detach().numpy())
        else:
            emb_view = lil_matrix(emb_all.detach().numpy())

        delta = 0.375 * 2
        matrix = generate_new_adj(emb_view.todense(), delta).astype(np.int16)

        # 将 NumPy 数组转换为 PyTorch 张量，并将其移动到 GPU
        tensor = torch.tensor(matrix.todense(), dtype=torch.int32, device=emb_all.device)

        left_matrix = tensor[:self.set_a_num, :self.set_a_num]

        max_count = int(self.set_a_num * self.set_a_num * self.view_relate_rate)
        flattened_matrix = left_matrix.flatten()
        top_k_values, top_k_indices_l = torch.topk(flattened_matrix,
                                                   k=min(max_count, torch.sum(left_matrix > 0)))
        # top_k_row_indices_l = top_k_indices_l // left_matrix.shape[-1]
        top_k_row_indices_l = torch.div(top_k_indices_l, left_matrix.shape[-1], rounding_mode='floor')
        # top_k_row_indices_l = torch.div(top_k_indices_l, left_matrix.shape[-1])
        top_k_col_indices_l = top_k_indices_l % left_matrix.shape[-1]
        left_edge_index = torch.cat([top_k_row_indices_l, top_k_col_indices_l], dim=-1)
        left_edge_index = left_edge_index.to(self.device)
        if left_edge_index.dim() != 2:
            left_edge_index = left_edge_index.view(2, -1)

        right_matrix = tensor[self.set_a_num:, self.set_a_num:]
        max_count = int(self.set_b_num * self.set_b_num * self.view_relate_rate)
        flattened_matrix_r = right_matrix.flatten()
        top_k_values_r, top_k_indices_r = torch.topk(flattened_matrix_r,
                                                     k=min(max_count, torch.sum(right_matrix > 0)))
        # top_k_row_indices_r = top_k_indices_r // right_matrix.shape[-1]
        top_k_row_indices_r = torch.div(top_k_indices_r, right_matrix.shape[-1], rounding_mode='floor')
        # top_k_row_indices_r = torch.div(top_k_indices_r, right_matrix.shape[-1])
        top_k_col_indices_r = top_k_indices_r % right_matrix.shape[-1]
        right_edge_index = torch.cat([top_k_row_indices_r, top_k_col_indices_r], dim=-1)
        right_edge_index = right_edge_index.to(self.device)
        if right_edge_index.dim() != 2:
            right_edge_index = right_edge_index.view(2, -1)

        left_emb_view = self.multi_left_view_model(emb_all[:self.set_a_num], left_edge_index)

        right_emb_view = self.multi_left_view_model(emb_all[self.set_a_num:], right_edge_index)

        emb_view = torch.cat([left_emb_view, right_emb_view], dim=0)
        return emb_view


    def forward(self, edge_lists):
        embedding_a, embedding_b = self.get_embeddings()   #特征增强------WL算法
        emb_all = torch.cat([embedding_a, embedding_b], dim=0).to(self.device)
        emb_view = self.multi_view(emb_all)    #同类型节点特征构造------multi-view
        emb_view_a = emb_view[:self.set_a_num]
        emb_view_b = emb_view[self.set_a_num:]
        embedding_a1, embedding_b1 = self.gnn1(embedding_a, embedding_b)
        embedding_a2, embedding_b2 = self.gnn2(embedding_a, embedding_b)

        embedding_a = torch.cat([embedding_a1, emb_view_a], dim=1).to(self.device)
        embedding_b = torch.cat([embedding_b1, emb_view_b], dim=1).to(self.device)
        # alpha = 0.8
        # embedding_a = alpha * embedding_a1 + (1 - alpha) * emb_view_a
        # embedding_b = alpha * embedding_b1 + (1 - alpha) * emb_view_b

        y = torch.einsum("ij, ij->i", [embedding_a[edge_lists[:, 0]], embedding_b[edge_lists[:, 1]]])
        y = torch.sigmoid(y)

        return y, embedding_a1, embedding_b1, embedding_a2, embedding_b2

    def multi_gnn_loss(self, embedding_a, embedding_b, embedding_a2, embedding_b2, adjacency_matrix1, gnn_loss_rate):
        emb1 = torch.cat([embedding_a, embedding_b], dim=0).to(self.device)
        emb2 = torch.cat([embedding_a2, embedding_b2], dim=0).to(self.device)

        loss_1 = F.kl_div(F.log_softmax(emb1, dim=1), F.softmax(emb2, dim=1))

        top_values, top_indices = torch.topk(adjacency_matrix1, k=2, dim=0)

        max_value = torch.max(top_values)
        min_value = torch.min(top_values)
        # 计算相似度损失
        adjacency_loss = 0.0
        for i in range(top_values.shape[0]):  # 遍历每一列
            for j in range(top_values.shape[1]):  # 遍历前十个最大值的索引
                # weight = torch.log(1 + top_values[i, j])
                weight = top_values[i, j]
                # adjacency_loss += 1 / (weight + 1) * similarity_loss(emb1[j], emb1[top_indices[i, j]])
                adjacency_loss += (1 + (weight - min_value) / (max_value - min_value)) \
                    * similarity_loss(emb1[j], emb1[top_indices[i, j]])
        adjacency_loss = adjacency_loss / (top_values.shape[0] * top_values.shape[1])  # to mean

        loss = gnn_loss_rate * loss_1 + (1 - gnn_loss_rate) * adjacency_loss

        return loss

    def loss(self, pred_y, y):
        assert y.min() >= 0, 'must 0~1'
        assert pred_y.size() == y.size(), 'must be same length'
        pos_ratio = y.sum() / y.size()[0]
        weight = torch.where(y > 0.5, 1. / pos_ratio, 1. / (1 - pos_ratio))
        # weight = torch.where(y > 0.5, (1-pos_ratio), pos_ratio)
        # criterion = FocalLoss()
        return F.binary_cross_entropy(pred_y, y, weight=weight)
