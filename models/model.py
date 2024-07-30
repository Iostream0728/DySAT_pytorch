# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/02/19 21:10:00
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss

from models.layers import StructuralAttentionLayer, TemporalAttentionLayer
from utils.utilities import fixed_unigram_candidate_sampler

class DySAT(nn.Module):
    def __init__(self, args, num_features, time_length): #初始化DySAT模型 self代表类的实例对象
        """[summary]

        Args:
            args：包含模型超参数和配置的对象
            time_length (int): 数据集中的总时间步数
            num_features(int):输入特征的维度
        """
        super(DySAT, self).__init__() #初始化父类
        self.args = args
        #args.window指定时间窗口的大小
        if args.window < 0: #小于0时，使用整个时间长度
            self.num_time_steps = time_length
        else:
            self.num_time_steps = min(time_length, args.window + 1)  # window = 0 => only self.
        self.num_features = num_features

        #结构注意力头和层的配置
        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        #时间注意力头和层的配置
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        #时间和空间注意力的丢弃率
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop

        #构建空间和时间注意力模块
        self.structural_attn, self.temporal_attn = self.build_model()

        #使用二元交叉熵损失函数
        self.bceloss = BCEWithLogitsLoss()

    def forward(self, graphs):
        #前向传播=> 指模型从输入特征到输出节点表示的过程
        #结构注意力前向传播
        structural_out = [] #初始化一个空列表 用于存储每个时间步的结构注意力输出
        for t in range(0, self.num_time_steps): #遍历所有时间步
            structural_out.append(self.structural_attn(graphs[t]))
        structural_outputs = [g.x[:,None,:] for g in structural_out] # list of [Ni, 1, F] #Ni表示第i个时间步的节点数量 F是特征维度

        # 填充结构注意力输出（将每个时间步的节点特征矩阵填充到相同的大小 即按照最大节点数来填充）
        maximum_node_num = structural_outputs[-1].shape[0] #获取最后一个时间步的节点数量，作为填充的最大节点数
        out_dim = structural_outputs[-1].shape[-1] #获取输出特征的维度
        structural_outputs_padded = [] #初始化一个空列表 存储填充后的结构注意力输出
        for out in structural_outputs:
            zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)
        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # 得到形状为[N, T, F]的张量，N是最大节点数，T是时间步，F是特征维度 
        
        # 时间注意力前向传播
        temporal_out = self.temporal_attn(structural_outputs_padded)
        
        return temporal_out

    def build_model(self):
        input_dim = self.num_features #input_dim初始化为输入特征的维度，即每个节点的特征数

        # 1: 结构注意力层
        structural_attention_layers = nn.Sequential() #structural_attention_layers是一个顺序容器 用于存储多个结构注意力层
        for i in range(len(self.structural_layer_config)): 
            layer = StructuralAttentionLayer(input_dim=input_dim, #输入特征的维度
                                             output_dim=self.structural_layer_config[i], #该层的输出特征维度
                                             n_heads=self.structural_head_config[i], #注意力头的数量
                                             attn_drop=self.spatial_drop, #丢弃率
                                             ffd_drop=self.spatial_drop, #丢弃率
                                             residual=self.args.residual '''是否使用残差连接''')
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer) 
            input_dim = self.structural_layer_config[i] #更新输出特征维度
        
        # 2: 时间注意力层
        input_dim = self.structural_layer_config[-1] #更新为最后一个结构注意力层的输出特征维度
        temporal_attention_layers = nn.Sequential() 
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]

        return structural_attention_layers, temporal_attention_layers #返回建构好的注意力层 在模型的前向传播中使用
        
    #计算模型的损失函数 用于衡量模型的预测结果与实际标签之间的差距
    def get_loss(self, feed_dict):
        node_1, node_2, node_2_negative, graphs = feed_dict.values() 
        #node_1,node_2为正样本节点对 node_2_negative为负样本节点对
        # run gnn
        final_emb = self.forward(graphs) # [N, T, F]
        self.graph_loss = 0 #初始化损失
        for t in range(self.num_time_steps - 1): #遍历时间步
            emb_t = final_emb[:, t, :].squeeze() #[N, F] 获取第t个时间的节点嵌入emb_t 其形状为[N,F]
            #使用squeeze去除不必要的维度
            source_node_emb = emb_t[node_1[t]] #正样本中的源节点嵌入
            tart_node_pos_emb = emb_t[node_2[t]] #正样本中的目标节点嵌入
            tart_node_neg_emb = emb_t[node_2_negative[t]] #负样本中的目标节点嵌入
            pos_score = torch.sum(source_node_emb*tart_node_pos_emb, dim=1) #正样本得分
            neg_score = -torch.sum(source_node_emb[:, None, :]*tart_node_neg_emb, dim=2).flatten() #负样本得分
            pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score)) #正样本损失
            neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score)) #负样本损失
            graphloss = pos_loss + self.args.neg_weight*neg_loss #总损失(负样本损失需要加权)
            self.graph_loss += graphloss #每个时间步的损失积累到self.graph_loss
        return self.graph_loss

            




