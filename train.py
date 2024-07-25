# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/02/20 10:25:13
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import argparse
import networkx as nx
import numpy as np
import dill
import pickle as pkl
import scipy
from torch.utils.data import DataLoader

from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data
from utils.minibatch import  MyDataset
from utils.utilities import to_device
from eval.link_prediction import evaluate_classifier
from models.model import DySAT

import torch
torch.autograd.set_detect_anomaly(True) #PyTorch中用于启用异常检测的函数

def inductive_graph(graph_former, graph_later):
    """创建 adj_train，使其包含来自 (t+1) 的节点，但仅包含来自 t 的边：这是为了进行归纳测试。

    参数：
    graph_former (networkx.Graph): 包含来自时间 t 的边的图。
    graph_later (networkx.Graph): 包含来自时间 t+1 的节点的图。
    """
    newG = nx.MultiGraph()
    newG.add_nodes_from(graph_later.nodes(data=True))
    newG.add_edges_from(graph_former.edges(data=False))
    return newG    #采用新的节点信息和旧的边信息组成一张新图 用于模型的归纳性测试


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #总时间步数，用于训练 评估和测试
    parser.add_argument('--time_steps', type=int, nargs='?', default=16,
                        help="total time steps used for train, eval and test")
    # 实验设置
    parser.add_argument('--dataset', type=str, nargs='?', default='Enron',
                        help='dataset name') #数据集名称
    parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)') #使用的GPU ID
    parser.add_argument('--epochs', type=int, nargs='?', default=200,
                        help='# epochs') #训练的轮数
    parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)') #验证频率（以轮数为单位）
    parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                        help='Testing frequency (in epochs)') #测试频率（轮数为单位）
    parser.add_argument('--batch_size', type=int, nargs='?', default=512,
                        help='Batch size (# nodes)') #批处理大小（节点数）
    parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                    help='True if one-hot encoding.') #是否使用one-hot编码
    parser.add_argument("--early_stop", type=int, default=10,
                        help="patient") #提前停止的耐心度
    # 单热编码（onehot编码）作为稀疏矩阵输入 - 因此对于大数据集没有扩展性问题。
    # 可调节的超参数
    # TODO: 实现尚未经过验证，性能可能不好。
    parser.add_argument('--residual', type=bool, nargs='?', default=True,
                        help='Use residual') #是否使用残差连接
    # 每个正样本对所对应的负样本数量
    parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
                        help='# negative samples per positive') 
    # 随机游走采样的步长
    parser.add_argument('--walk_len', type=int, nargs='?', default=20,
                        help='Walk length for random walk sampling')
    
    parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                        help='Weightage for negative samples') # 二值交叉熵损失函数中负样本的权重
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                        help='Initial learning rate for self-attention model.') #自主意力模型的初始学习率
    parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                        help='Spatial (structural) attention Dropout (1 - keep probability).') #空间（结构）注意力的Dropout（1 - 保持概率）
    parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        help='Temporal attention Dropout (1 - keep probability).') #时间注意力的Dropout（1 - 保持概率）
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.') #自注意力模型的权重衰减
    # 架构参数
    parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
                        help='Encoder layer config: # attention heads in each GAT layer') #每层GAT层中的注意力头数量配置
    parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each GAT layer') #每层GAT层中的单元数配置
    parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # attention heads in each Temporal layer') #每层时间层中的注意力头数量配置
    parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each Temporal layer') #每层时间层中的单元数配置
    parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward') #是否使用位置前馈网络
    parser.add_argument('--window', type=int, nargs='?', default=-1,
                        help='Window for temporal attention (default : -1 => full)') #时间注意力的窗口大小（默认：-1，表示全部）
    args = parser.parse_args() #args用于存储上面定义的参数
    print(args)

    #graphs, feats, adjs = load_graphs(args.dataset)
    graphs, adjs = load_graphs(args.dataset)
    if args.featureless == True:
        feats = [scipy.sparse.identity(adjs[args.time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if
             x.shape[0] <= adjs[args.time_steps - 1].shape[0]] #生成一个one-hot特征矩阵feats

    assert args.time_steps <= len(adjs), "Time steps is illegal" #检查时间步是否合法

    context_pairs_train = get_context_pairs(graphs, adjs) #基于加载的图数据和邻接矩阵获取训练用的上下文对（指图中两个节点的组合）
    #用于捕捉节点之间的关系

    # 为link prediction加载评估数据
    train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
        test_edges_pos, test_edges_neg = get_evaluation_data(graphs)
    '''train_edges_pos：训练集中的正样本边。
    train_edges_neg：训练集中的负样本边。
    val_edges_pos：验证集中的正样本边。
    val_edges_neg：验证集中的负样本边。
    test_edges_pos：测试集中的正样本边。
    test_edges_neg：测试集中的负样本边。'''
    
    print("No. Train: Pos={}, Neg={} \nNo. Val: Pos={}, Neg={} \nNo. Test: Pos={}, Neg={}".format(
        len(train_edges_pos), len(train_edges_neg), len(val_edges_pos), len(val_edges_neg),
        len(test_edges_pos), len(test_edges_neg))) #用于输出训练集、验证集和测试集中正样本和负样本的数量

    # Create the adj_train so that it includes nodes from (t+1) but only edges from t: this is for the purpose of
    # inductive testing.
    new_G = inductive_graph(graphs[args.time_steps-2], graphs[args.time_steps-1])
    graphs[args.time_steps-1] = new_G
    adjs[args.time_steps-1] = nx.adjacency_matrix(new_G)

    # build dataloader and model
    device = torch.device("cuda:0")
    dataset = MyDataset(args, graphs, feats, adjs, context_pairs_train)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=10, 
                            collate_fn=MyDataset.collate_fn)
    #dataloader = NodeMinibatchIterator(args, graphs, feats, adjs, context_pairs_train, device) 
    model = DySAT(args, feats[0].shape[1], args.time_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # in training
    best_epoch_val = 0
    patient = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        for idx, feed_dict in enumerate(dataloader):
            feed_dict = to_device(feed_dict, device)
            opt.zero_grad()
            loss = model.get_loss(feed_dict)
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())

        model.eval()
        emb = model(feed_dict["graphs"])[:, -2, :].detach().cpu().numpy()
        val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                            train_edges_neg,
                                                            val_edges_pos, 
                                                            val_edges_neg, 
                                                            test_edges_pos,
                                                            test_edges_neg, 
                                                            emb, 
                                                            emb)
        epoch_auc_val = val_results["HAD"][1]
        epoch_auc_test = test_results["HAD"][1]

        if epoch_auc_val > best_epoch_val:
            best_epoch_val = epoch_auc_val
            torch.save(model.state_dict(), "./model_checkpoints/model.pt")
            patient = 0
        else:
            patient += 1
            if patient > args.early_stop:
                break

        print("Epoch {:<3},  Loss = {:.3f}, Val AUC {:.3f} Test AUC {:.3f}".format(epoch, 
                                                                np.mean(epoch_loss),
                                                                epoch_auc_val, 
                                                                epoch_auc_test))
    # Test Best Model
    model.load_state_dict(torch.load("./model_checkpoints/model.pt"))
    model.eval()
    emb = model(feed_dict["graphs"])[:, -2, :].detach().cpu().numpy()
    val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                        train_edges_neg,
                                                        val_edges_pos, 
                                                        val_edges_neg, 
                                                        test_edges_pos,
                                                        test_edges_neg, 
                                                        emb, 
                                                        emb)
    auc_val = val_results["HAD"][1]
    auc_test = test_results["HAD"][1]
    print("Best Test AUC = {:.3f}".format(auc_test))


                







