import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

import dgl.function as fn
from common import STDLayerNorm as LayerNorm
from dgl.nn.pytorch.softmax import edge_softmax
from dgl import DGLGraph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Transition = namedtuple('Transition', ('node_state', 'node_action', 'node_prob', 'struct_state', 'struct_action', 'struct_prob', 'reward'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)
    

class PPO(nn.Module):
    def __init__(self, input_dim, output_dim, device, eps_clip=0.2):
        super(PPO, self).__init__()
        self.memory = Memory()
        self.eps_clip = eps_clip
        self.device = device
        
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc_pi = nn.Linear(64, output_dim)
        
        self.s1 = nn.Linear(input_dim + 2, 64)
        self.s2 = nn.Linear(64, 32)
        self.s3 = nn.Linear(32, output_dim)
        
        self.layer_norm(self.fc1, std=1.0)
        self.layer_norm(self.fc2, std=1.0)
        self.layer_norm(self.fc_pi, std=0.01)
        
        self.layer_norm(self.s1, std=1.0)
        self.layer_norm(self.s2, std=1.0)
        self.layer_norm(self.s3, std=0.01)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        
    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    
    def pi(self, x):     #接受输入 x，通过多个全连接层和激活函数的处理，最终输出一个表示概率分布的张量 prob，用于表示模型对每个类别的预测概率
        x = x.to(device)
        x = self.fc1(x)
        x = torch.tanh(x)

        x = self.fc2(x)
        x = torch.tanh(x)
        
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=1)
        
        return prob

    def pi_s(self, x):  
        x=self.s1(x)
        x = torch.tanh(x)

        x=self.s2(x)
        x = torch.tanh(x)
        
        x = self.s3(x)
        prob = F.softmax(x, dim=1)
        
        return prob
    
    def put_data(self, x):
        node_prob = x[2].gather(dim=1, index=torch.unsqueeze(x[1], dim=1))
        struct_prob = x[5].gather(dim=1, index=torch.unsqueeze(x[4], dim=1))
        self.memory.push(x[0], torch.unsqueeze(x[1], dim=1), node_prob, x[3], torch.unsqueeze(x[4], dim=1), struct_prob, x[6])

    def train_net(self, epochs=5, batch_size=256):
        batch = self.memory.sample()
        node_state, node_action, node_old_prob, struct_state, struct_action, struct_old_prob, reward = torch.cat(batch.node_state), torch.cat(batch.node_action), torch.cat(batch.node_prob), torch.cat(batch.struct_state), torch.cat(batch.struct_action), torch.cat(batch.struct_prob), torch.cat(batch.reward)
        
        # sample from memory
        idx = np.random.choice(node_state.size()[0], min(node_state.size()[0], batch_size), replace=False)
        node_state, node_action, node_old_prob, struct_state, struct_action, struct_old_prob, reward = node_state[idx], node_action[idx], node_old_prob[idx], struct_state[idx], struct_action[idx], struct_old_prob[idx], reward[idx]

        for i in range(epochs):
            advantage = reward
            node_new_prob = self.pi(node_state).gather(dim=1, index=node_action)
            struct_new_prob = self.pi_s(struct_state).gather(dim=1, index=struct_action)
            
            ratio1 = torch.exp(torch.log(node_new_prob) - torch.log(node_old_prob))
            ratio2 = torch.exp(torch.log(struct_new_prob) - torch.log(struct_old_prob))
            
            surr1 = ratio1 * advantage
            surr2 = torch.clamp(ratio1, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss1 = -torch.min(surr1, surr2).mean()
            
            surr3 = ratio2 * advantage
            surr4 = torch.clamp(ratio2, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss2 = -torch.min(surr3, surr4).mean()
            
            loss = loss1 + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.lr_scheduler.step()

        self.memory = Memory()


class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x

class PositionwiseFeedForward(nn.Module):   #定义了一个简单的位置前馈神经网络层，它包含了两个线性变换层和一个丢弃层，并通过 ReLU 激活函数实现非线性变换。这样的 FFN 组件可以用于图神经网络中的每个层，以对输入特征进行非线性映射和变换
    "Implements FFN equation."
    def __init__(self, model_dim, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, d_hidden)
        self.w_2 = nn.Linear(d_hidden, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

    def init(self):
        gain = nn.init.calculate_gain('relu')  #计算了 ReLU 激活函数的增益值
        nn.init.xavier_normal_(self.w_1.weight, gain=gain)   #使用 Xavier 正态分布初始化方法来初始化 self.w_1  self.w_2 的权重。同样，gain 参数指定了初始化的增益值
        nn.init.xavier_normal_(self.w_2.weight, gain=gain)


class MAGNALayer(nn.Module):
    def __init__(self,
                 in_feats: int,
                 hidden_dim: int,
                 num_heads: int,
                 alpha,
                 hop_num,
                 feat_drop,
                 attn_drop,
                 topk_type='local',
                 top_k=-1,
                 layer_norm=True,
                 feed_forward=True,
                 head_tail_shared=True,
                 negative_slope=0.2):
        """
        """
        super(MAGNALayer, self).__init__()
        self.topk_type = topk_type
        self._in_feats = in_feats
        self._out_feats = hidden_dim
        self._num_heads = num_heads
        self.alpha = alpha
        self.hop_num = hop_num
        self.top_k = top_k ## FOR dense graph, edge selection
        self.head_tail_shared = head_tail_shared
        self.layer_norm = layer_norm
        self.feed_forward = feed_forward
        self._att_dim = hidden_dim // num_heads

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if self.head_tail_shared:   #是否共享头节点和尾节点的线性变换权重
            self.fc = nn.Linear(in_feats, self._out_feats, bias=False) .to(device)  #创建一个线性变换层，并将其赋值给 self.fc。这表示头节点和尾节点的线性变换权重是共享的，它们使用同一个线性变换层进行映射
        else:
            self.fc_head = nn.Linear(in_feats, self._out_feats, bias=False) .to(device) #创建一个线性变换层，并将其赋值给 self.fc_head。这表示头节点和尾节点的线性变换权重是分别独立的，它们使用两个不同的线性变换层进行映射
            self.fc_tail = nn.Linear(in_feats, self._out_feats, bias=False) .to(device) #在这个分支中，同样创建一个线性变换层，并将其赋值给 self.fc_tail
            self.fc = nn.Linear(in_feats, self._out_feats, bias=False) .to(device)  #最后，无论头尾节点是否共享权重，都创建一个线性变换层，并将其赋值给 self.fc。这个变换层用于其他情况，即头节点和尾节点的线性变换权重是相同的。
        self.fc_out = nn.Linear(self._out_feats, self._out_feats, bias=False).to(device)
        #可能用于在图神经网络中的某个阶段或输出层进行线性变换，以适应特定的任务需求或得到最终的预测结果。线性变换层可以将输入特征进行加权线性组合，从而产生输出特征，通常在图神经网络中用于节点分类、图分类等任务
        if in_feats != self._out_feats:  #这样的设计通常用于残差连接或跳跃连接等情况，以帮助网络更好地学习特征表示
            self.res_fc = nn.Linear(in_feats, self._out_feats, bias=False).to(device)  #self.res_fc：这是类的一个属性，用于存储创建的线性变换层或标识变换层。要对输入特征进行线性映射，使得输入特征和目标特征的维度一致。
        else:
            self.res_fc = Identity()   #Identity(): 这是一个标识变换层，它不进行任何变换，仅将输入特征原样输出

        self.attn_h = nn.Parameter(torch.FloatTensor(size=(1, self._num_heads, self._att_dim)), requires_grad=True)  #创建一个形状为 (1, self._num_heads, self._att_dim) 的浮点张量（FloatTensor）。这个张量用于存储注意力机制中的头部信息
        #requires_grad=True: 这是一个标志，表示创建的参数需要进行梯度计算（可训练）。默认情况下，requires_grad 的值为 True，表示创建的参数需要梯度
        self.attn_t = nn.Parameter(torch.FloatTensor(size=(1, self._num_heads, self._att_dim)), requires_grad=True)
        self.graph_norm = LayerNorm(num_features=in_feats)  # entity feature normalization,层归一化是对每个样本在特征维度上进行独立的归一化,一个层归一化层 self.graph_norm 被创建，并用于对输入数据进行归一化处理
        self.feed_forward = PositionwiseFeedForward(model_dim=self._out_feats, d_hidden=4 * self._out_feats)  # entity feed forward  位置前馈神经网络将输入特征映射到更高维度的隐藏层，然后再映射回原始的输入特征维度。这样的设计允许网络进行非线性变换，从而更好地学习复杂的特征表示
        self.ff_norm = LayerNorm(num_features=self._out_feats)  # entity feed forward normalization  ，一个层归一化层 self.ff_norm 被创建，并用于对位置前馈神经网络的输出数据进行标准化处理。
        self.reset_parameters()
        self.attention_mask_value = -1e20
        #用于表示一个特殊的遮罩值，比如 -1e20。通过将这个值与注意力权重矩阵相乘，可以将那些应该被抑制的注意力权重设置为一个非常小的值，从而在计算过程中忽略这些权重，达到限制的效果

    def reset_parameters(self):   #模型中的权重参数被重新初始化为随机值，以便在训练过程中更好地学习数据的特征表示。特别是对于多层模型，重新初始化参数可以帮助模型更快地收敛和取得更好的训练效果。
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('tanh')
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight.data, gain=gain)
        nn.init.xavier_normal_(self.fc.weight.data, gain=gain)
        if not self.head_tail_shared:
            nn.init.xavier_normal_(self.fc_head.weight.data, gain=gain)
            nn.init.xavier_normal_(self.fc_tail.weight.data, gain=gain)
            nn.init.xavier_normal_(self.fc_out.weight.data, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)
        nn.init.xavier_normal_(self.attn_h, gain=gain)

    def forward(self, graph: DGLGraph, features, drop_edge_ids=None):
        ###Attention computation: pre-normalization structure
        features=torch.tensor(features)
        features=features.to(device)
        graph = graph.local_var()  #使用 local_var() 方法来创建一个图的本地版本。这样做的好处是可以在不改变原始图的情况下，进行临时的修改或存储，避免了对原始数据的影响
        if self.layer_norm:   #当开启了层归一化时，输入特征数据 features 会经过层归一化处理得到 h；而当未开启层归一化时，h 会直接等于输入特征数据 features，不进行任何额外处理
            h = self.graph_norm(features).to(device)
        else:
            h = features
        if self.head_tail_shared:
            feat = self.fc(self.feat_drop(h)).view(-1, self._num_heads, self._att_dim)
            feat_tanh = torch.tanh(feat)
            eh = (feat_tanh * self.attn_h).sum(dim=-1).unsqueeze(-1)
            et = (feat_tanh * self.attn_t).sum(dim=-1).unsqueeze(-1)
            graph.ndata.update({'ft': feat, 'eh': eh, 'et': et})
        else:
            feat_head = torch.tanh(self.fc_head(self.feat_drop(h))).view(-1, self._num_heads, self._att_dim) #计算了头部和尾部注意力的特征
            feat_tail = torch.tanh(self.fc_tail(self.feat_drop(h))).view(-1, self._num_heads, self._att_dim)
            feat = self.fc(self.feat_drop(h)).view(-1, self._num_heads, self._att_dim)# 计算特征feat
            eh = (feat_head * self.attn_h).sum(dim=-1).unsqueeze(-1)   #计算头部注意力  源节点的注意力权重
            et = (feat_tail * self.attn_t).sum(dim=-1).unsqueeze(-1)   #计算尾部注意力   这里的头部和尾部分别指什么？？？  源节点和目标节点
            graph.ndata.update({'ft': feat, 'eh': eh, 'et': et})
        graph.apply_edges(fn.u_add_v('eh', 'et', 'e'))
        #fn.u_add_v 表示对图的边的源节点和目标节点的特征进行逐元素相加操作，‘e’是边的特征数据  将源节点和目标节点的注意力权重进行逐元素相加，并将结果存储在边的特征数据中，用于对头部和尾部的注意力信息进行合并和融合
        attations = graph.edata.pop('e')   #边数据中的 'e' 这个键被移除，并将其对应的值存储在变量 attentions中
        attations = self.leaky_relu(attations)  #通过应用 Leaky ReLU（带泄漏的线性整流单元）函数激活 attentions 变量的值。
        if drop_edge_ids is not None:   #drop_edge_ids 是一个变量，用于指定在图神经网络中需要丢弃（忽略）的特定边的边ID  设置为一个特定的非常小的值相当于丢弃
            attations[drop_edge_ids] = self.attention_mask_value

        if self.top_k <= 0:   #指定每个节点保留的注意力权重的最大数量（top k），小于等于0，表示不进行限制，保留所有的注意力权重；如果大于0，则根据该值保留每个节点的前 k 个最大的注意力权重
            graph.edata['a'] = edge_softmax(graph, attations)
        else:
            if self.topk_type == 'local':
                graph.edata['e'] = attations
                attations = self.topk_attention(graph)
                graph.edata['a'] = edge_softmax(graph, attations)  ##return attention scores
            else:
                graph.edata['e'] = edge_softmax(graph, attations)
                graph.edata['a'] = self.topk_attention_softmax(graph)

        rst = self.ppr_estimation(graph=graph)
        rst = rst.flatten(1)   #对变量 rst 进行形状转换，将其转换为一个一维张量,如果 rst 张量的原始形状是 (batch_size, num_nodes, num_features)，那么经过 flatten(1) 后，它的形状将变为 (batch_size, num_nodes * num_features)
        rst = self.fc_out(rst)   #用于将高维的输入特征映射到更低维的空间，或者对输入特征进行降维、压缩等处理
        features=torch.tensor(features)
        resval = self.res_fc(features) .to(device)  #使用全连接层 self.res_fc 对输入特征 features 进行线性变换，并将结果赋值给变量 resval
        rst = resval + self.feat_drop(rst)   #这种操作通常用于引入残差连接（residual connection），即将某一层的输出与之前某一层的输出相加，从而在神经网络中引入跳跃连接，有助于提高梯度的传播和网络的训练效果
        if not self.feed_forward:   #是否使用前馈神经网络（Feed-Forward Network）。如果 self.feed_forward 为 True，则会执行前馈神经网络的操作；如果为 False，则会执行激活函数操作。
            return F.elu(rst)

        if self.layer_norm:
            rst_ff = self.feed_forward(self.ff_norm(rst))
        else:
            rst_ff = self.feed_forward(rst)
        rst = rst + self.feat_drop(rst_ff)
        return rst

    def ppr_estimation(self, graph: DGLGraph):   #使用图神经网络的消息传递过程迭代计算个性化PageRank，通过一系列的循环和节点特征的更新，得到最终的个性化PageRank估计结果
        graph = graph.local_var()   #创建了图的局部变量，目的是为了防止在后续的计算过程中对原始图数据造成修改
        feat_0 = graph.ndata.pop('ft')   #节点的初始特征，用于初始化个性化PageRank的计算   ft：feat
        feat = feat_0
        attentions = graph.edata.pop('a')   #a：边的注意力权重，用于进行消息传递计算
        for _ in range(self.hop_num):
            graph.ndata['h'] = feat
            graph.edata['a_temp'] = self.attn_drop(attentions)
            graph.update_all(fn.u_mul_e('h', 'a_temp', 'm'), fn.sum('m', 'h'))  #先节点特征 'h' 乘以边的属性 'a_temp' 得到消息 'm'，也就是将节点特征与边的注意力权重相乘得到消息m；再对收到的消息 'm' 进行求和，并将汇聚结果作为节点特征 'h' 更新节点的特征
            feat = graph.ndata.pop('h')
            feat = (1.0 - self.alpha) * feat + self.alpha * feat_0   #计算节点特征的更新，其中 self.alpha 是一个控制衰减程度的参数，用于在迭代过程中融合原始节点特征 feat_0 和新的节点特征 feat
            feat = self.feat_drop(feat)
        return feat
    #  根据输入图和特征，使用Personalized PageRank算法对节点进行特征表示的估计。
    #  每次迭代中，通过更新邻居节点的特征信息，逐步传播并整合节点的上下文信息，最终得到一个代表节点特征的向量。

    def topk_attention(self, graph: DGLGraph):
        graph = graph.local_var()# the graph should be added a self-loop edge
        def send_edge_message(edges):     #定义了一个 send_edge_message 函数，它将边数据 'e' 作为消息进行发送。这意味着边的注意力权重将作为消息从边传递给节点
            return {'m_e': edges.data['e']}
        def topk_attn_reduce_func(nodes):   #将边的注意力权重进行排序，并选择其中前 top_k 个最大的权重值。然后将选中的最大权重值作为节点的属性 'kth_e' 返回
            topk = self.top_k
            attentions = nodes.mailbox['m_e']
            neighbor_num = attentions.shape[1]
            if topk > neighbor_num:
                topk = neighbor_num
            topk_atts, _ = torch.topk(attentions, k=topk, dim=1)
            kth_attn_value = topk_atts[:, topk-1]
            return {'kth_e': kth_attn_value}
        # 在给定的图中计算每个节点的前k个最大关注度值。
        # 它首先通过边上的关注度值发送消息，然后在节点上进行聚合，并返回每个节点的第k个最大关注度值。

        graph.register_reduce_func(topk_attn_reduce_func)
        graph.register_message_func(send_edge_message)
        graph.update_all(message_func=send_edge_message, reduce_func=topk_attn_reduce_func)
        def edge_score_update(edges):
            scores, kth_score = edges.data['e'], edges.dst['kth_e']
            scores[scores < kth_score] = self.attention_mask_value
            return {'e': scores}
        graph.apply_edges(edge_score_update)
        topk_attentions = graph.edata.pop('e')
        return topk_attentions
    # 该段代码对图进行消息传递和聚合操作，
    # 然后根据目标节点的第k个最大关注度值更新边的关注度值，并返回每条边的前k个最大关注度值。

    def topk_attention_softmax(self, graph: DGLGraph):
        graph = graph.local_var()
        def send_edge_message(edges):
            return {'m_e': edges.data['e'], 'm_e_id': edges.data['e_id']}
        def topk_attn_reduce_func(nodes):
            topk = self.top_k
            attentions = nodes.mailbox['m_e']
            edge_ids = nodes.mailbox['m_e_id']
            topk_edge_ids = torch.full(size=(edge_ids.shape[0], topk), fill_value=-1, dtype=torch.long)
            if torch.cuda.is_available():
                topk_edge_ids = topk_edge_ids.cuda()
            attentions_sum = attentions.sum(dim=2)   #对边注意力权重张量进行求和操作，将边的注意力权重按照维度2（第3个维度，从0开始计数）进行求和，得到节点收到的所有边的注意力权重之和
            neighbor_num = attentions_sum.shape[1]
            if topk > neighbor_num:
                topk = neighbor_num
            topk_atts, top_k_neighbor_idx = torch.topk(attentions_sum, k=topk, dim=1)     #对求和后的边注意力权重进行排序操作，选择前 top_k 个最大的权重值，并得到它们的索引
            top_k_neighbor_idx = top_k_neighbor_idx.squeeze(dim=-1)   #对索引张量进行维度压缩操作，将维度 -1（最后一个维度）进行压缩，从而得到一维的索引张量
            row_idxes = torch.arange(0, top_k_neighbor_idx.shape[0]).view(-1,1)  #一个创建张量的操作，用于生成行索引，从0到节点收到的边的数量
            top_k_attention = attentions[row_idxes, top_k_neighbor_idx]    #通过索引张量从边注意力权重张量中选择节点收到的前 top_k 个最大边的注意力权重
            top_k_edge_ids = edge_ids[row_idxes, top_k_neighbor_idx]
            top_k_attention_norm = top_k_attention.sum(dim=1)   #对前 top_k 个最大边的注意力权重进行求和操作，得到节点收到的前 top_k 个最大边的注意力权重之和
            topk_edge_ids[:, torch.arange(0,topk)] = top_k_edge_ids   #将选择的前 top_k 个最大边的标识符放入之前创建的 topk_edge_ids 张量中
            return {'topk_eid': topk_edge_ids, 'topk_norm': top_k_attention_norm}   #这样的计算通常用于设计更复杂的图注意力机制或图卷积操作，以选择节点关联的最重要的邻居
        graph.register_reduce_func(topk_attn_reduce_func)  #在消息汇聚阶段，每个节点接收从其相邻节点发送过来的消息，并通过自定义的函数对这些消息进行汇聚，得到节点的新状态或特征
        graph.register_message_func(send_edge_message)  #在消息发送阶段，每条边根据自定义的函数将消息从源节点发送到目标节点
        graph.update_all(message_func=send_edge_message, reduce_func=topk_attn_reduce_func)
        topk_edge_ids = graph.ndata['topk_eid'].flatten()
        topk_edge_ids = topk_edge_ids[topk_edge_ids >=0]
        mask_edges = torch.zeros((graph.number_of_edges(), 1))
        if torch.cuda.is_available():
            mask_edges = mask_edges.cuda()
        mask_edges[topk_edge_ids] = 1
        attentions = graph.edata['e'].squeeze(dim=-1)
        attentions = attentions * mask_edges
        graph.edata['e'] = attentions.unsqueeze(dim=-1)
        def edge_score_update(edges):
            scores = edges.data['e']/edges.dst['topk_norm']
            return {'e': scores}
        graph.apply_edges(edge_score_update)
        topk_attentions = graph.edata.pop('e')
        return topk_attentions
