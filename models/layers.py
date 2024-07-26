# -*- encoding: utf-8 -*-
'''
@File    :   layers.py
@Time    :   2021/02/18 14:30:13
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_scatter import scatter

import copy


class StructuralAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                output_dim, 
                n_heads, 
                attn_drop, 
                ffd_drop,
                residual):
        super(StructuralAttentionLayer, self).__init__() #调用父类的'nn.Module'的初始化函数
        #确保'StructuralAttentionLayer'正确继承PyTorch的'nn.Module'基础功能
        self.out_dim = output_dim // n_heads #每个注意力头的输出维度
        self.n_heads = n_heads
        self.act = nn.ELU() #激活函数

        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False) #一个线性变换层
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim)) #左侧的注意力参数
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim)) #右侧的注意力参数
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2) #LeakyReLU激活函数

        self.attn_drop = nn.Dropout(attn_drop) #注意力机制的dropout
        self.ffd_drop = nn.Dropout(ffd_drop) #前馈网络的dropout

        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)

        #调用 Xavier 初始化方法，用于初始化模型的权重。
        #Xavier 初始化有助于保持激活值和梯度的方差一致，从而加速训练并改善模型性能。
        self.xavier_init()

    #实现图注意力机制的计算
    def forward(self, graph):
        graph = copy.deepcopy(graph) #对输入图进行深拷贝，以防止修改原始图数据
        edge_index = graph.edge_index #图的边索引
        edge_weight = graph.edge_weight.reshape(-1, 1) #边的权重 并调整形状
        H, C = self.n_heads, self.out_dim
        x = self.lin(graph.x).view(-1, H, C) # [N, heads, out_dim]
        # attention
        alpha_l = (x * self.att_l).sum(dim=-1).squeeze() # [N, heads]
        alpha_r = (x * self.att_r).sum(dim=-1).squeeze()
        alpha_l = alpha_l[edge_index[0]] # [num_edges, heads] 提取边的起始节点的注意力权重
        alpha_r = alpha_r[edge_index[1]] #提取边的中止节点的注意力权重
        alpha = alpha_r + alpha_l 
        alpha = edge_weight * alpha
        alpha = self.leaky_relu(alpha)
        coefficients = softmax(alpha, edge_index[1]) # [num_edges, heads] 对注意力权重进行softmax操作，得到每条边的归一化注意力系数
    

        # dropout
        if self.training:
            coefficients = self.attn_drop(coefficients)
            x = self.ffd_drop(x)
        x_j = x[edge_index[0]]  # [num_edges, heads, out_dim] 提取每条边起始节点的特征

        # 输出
        out = self.act(scatter(x_j * coefficients[:, :, None], edge_index[1], dim=0, reduce="sum")) #根据注意力系数将起始节点的特征聚合到终止节点
        out = out.reshape(-1, self.n_heads*self.out_dim) #[num_nodes, output_dim] 
        if self.residual: #如果启用了残差连接
            out = out + self.lin_residual(graph.x) #将原始输入特征与输出特征相加
        graph.x = out #将更新后的节点特征赋值回‘graph.x’
        return graph

    def xavier_init(self): #初始化StructuralAttentionLayer类中注意力参数的权重
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

        
class TemporalAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                n_heads, 
                num_time_steps, 
                attn_drop, 
                residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # 定义权重（都可训练）
        #时间步位置嵌入，用于添加时间步信息，大小为'[num_time_steps, input_dim]'
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        #查询权重矩阵，大小为'[input_dim, input_dim]'
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        #键权重矩阵，大小为'[input_dim, input_dim]'
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        #值权重矩阵，大小为'[input_dim, input_dim]'
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # 前馈网络
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 随机丢弃部分注意力权重
        self.attn_dp = nn.Dropout(attn_drop)
        #调用 xavier_init 方法对上述权重参数进行 Xavier 均匀分布初始化。
        self.xavier_init()


    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: 将位置嵌入添加到输入数据中，用于捕捉时间步的信息：
        position_inputs = torch.arange(0,self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(inputs.device)
        temporal_inputs = inputs + self.position_embeddings[position_inputs] # [N, T, F]

        # 2: 多头注意力的查询 键和值
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]

        # 3: 拆分 连接和缩放
        split_size = int(q.shape[-1]/self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        
        outputs = torch.matmul(q_, k_.permute(0,2,1)) # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)
       
        # 4: 计算注意力权重（只考虑从前和当前的时间步）
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val) #创建下三角矩阵tril 用于掩蔽未来时间步的信息
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1) # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2**32+1) #将 outputs 中对应 masks 为 0 的元素填充为一个非常小的值 -2**32 + 1，从而使这些位置在 softmax 操作后接近于 0。
        outputs = torch.where(masks==0, padding, outputs) 
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs # [h*N, T, T]
                
        # 5: 对注意力权重进行部分丢弃/Dropout
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        #将这些小张量沿第一个维度（节点数）连接，生成形状为‘[N,T,F]’的输出张量
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F]
        
        # 6: 前馈网络和残差连接
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs
        
    #前馈方法（引入非线性变换来捕捉更复杂的特征和关系）
    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    #使用Xavier均匀分布初始化所有权重参数
    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)
