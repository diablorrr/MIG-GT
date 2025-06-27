# coding: utf-8

import os
import json

from mig_gt.configs.default_config import add_arguments_by_config_class, combine_args_into_config
from mig_gt.configs.mm_mgdcf_default_config import MMMGDCFConfig
from mig_gt.configs.masked_mm_mgdcf_default_config import load_masked_mm_mgdcf_default_config 
import sys
import argparse
import time
import torch.nn.functional as F

from mig_gt.layers.mirf_gt import MIGGT
from mig_gt.vector_search.vector_search import VectorSearchEngine



# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='baby', help='name of datasets')
parser.add_argument('--method', type=str)
parser.add_argument('--result_dir', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--gpu', type=str)

# 根据配置类自动为参数解析器（parser）添加参数
config_class = MMMGDCFConfig
parser = add_arguments_by_config_class(parser, config_class)
args = parser.parse_args()

# 根据数据集名称，返回一个配置对象 -> 里面全是超参数的设置
config = load_masked_mm_mgdcf_default_config(args.dataset)
config = combine_args_into_config(config, args)

print(config)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


from mig_gt.utils.random_utils import reset_seed

# 固定随机数种子
reset_seed(args.seed)

from mig_gt.layers.mm_mgdcf import MMMGDCF
import shortuuid
from mig_gt.layers.sign import random_project, sign_pre_compute
from mig_gt.losses import compute_info_bpr_loss, compute_l2_loss
from mig_gt.utils.data_loader_utils import create_tensors_dataloader
from mig_gt.evaluation.ranking import evaluate_mean_global_metrics
from mig_gt.layers.mgdcf import MGDCF
from mig_gt.load_data import load_data
import torch
import numpy as np
import dgl
import dgl.function as fn
import time
import torch.nn as nn
from dataclasses import asdict



embedding_size = config.embedding_size 



device = "cuda"


# 加载和划分数据集：用户数量、物品数量、文本特征、图像特征、用户-物品交互边、每个用户的交互物品、需要屏蔽的交互
train_user_item_edges, valid_user_item_edges, test_user_item_edges, train_user_items_dict, train_mask_user_items_dict, valid_user_items_dict, valid_mask_user_items_dict, test_user_items_dict, test_mask_user_items_dict, num_users, num_items, v_feat, t_feat = load_data(args.dataset)

start_time = time.time()


run_id = shortuuid.uuid()

# 存放结果的目录
result_dir = args.result_dir

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

result_path = os.path.join(result_dir, "{}.json".format(run_id))
tmp_result_path = os.path.join(result_dir, "{}.json.tmp".format(run_id))



if config.use_rp:
    v_feat = random_project(v_feat, t_feat.size(-1))



num_train_user_item_edges = len(train_user_item_edges)
# 根据 用户数量、物品数量、用户-物品交互边 构造同构图（图中只有一种类型的节点和边：用户、物品被视为同一类型节点）
g = MGDCF.build_sorted_homo_graph(train_user_item_edges, num_users=num_users, num_items=num_items).to(device)
assert g.num_edges() == num_train_user_item_edges * 2 + num_users + num_items


num_nodes = g.num_nodes()

# 计算每个节点的入度：即每个用户/物品被连接的次数
degs = g.in_degrees().to(device)

# 初始化 物品-物品图
item_item_g = None



    

v_feat = v_feat.to(device)
t_feat = t_feat.to(device)



# 初始化 用户、物品的嵌入向量
user_embeddings = np.random.randn(num_users, embedding_size) / np.sqrt(embedding_size)
user_embeddings = torch.tensor(user_embeddings, dtype=torch.float32, requires_grad=True, device=device)
item_embeddings = np.random.randn(num_items, embedding_size) / np.sqrt(embedding_size)
item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32, requires_grad=True, device=device)


method = args.method

if method == "mig":

    model = MMMGDCF(
        k_e=config.k_e,
        k_t=config.k_t,
        k_v=config.k_v,

        alpha=config.alpha, 
        beta=config.beta, 

        input_feat_drop_rate=config.input_feat_drop_rate,
        feat_drop_rate=config.feat_drop_rate,
        user_x_drop_rate=config.user_x_drop_rate,
        item_x_drop_rate=config.item_x_drop_rate, 
        edge_drop_rate=config.edge_drop_rate, 
        z_drop_rate=config.z_drop_rate,
        item_v_in_channels=v_feat.size(-1),
        item_v_hidden_channels_list=[config.feat_hidden_units, embedding_size], 
        item_t_in_channels=t_feat.size(-1), 
        item_t_hidden_channels_list=[config.feat_hidden_units, embedding_size], 
        bn=config.bn,
    ).to(device)

elif method == "mig_gt":

    model = MIGGT(
        # k=config.k,

        k_e=config.k_e,
        k_t=config.k_t,
        k_v=config.k_v,

        alpha=config.alpha, 
        beta=config.beta, 

        input_feat_drop_rate=config.input_feat_drop_rate,
        feat_drop_rate=config.feat_drop_rate,
        user_x_drop_rate=config.user_x_drop_rate,
        item_x_drop_rate=config.item_x_drop_rate, 
        edge_drop_rate=config.edge_drop_rate, 
        z_drop_rate=config.z_drop_rate,
        user_in_channels=config.embedding_size,
        item_v_in_channels=v_feat.size(-1),
        item_v_hidden_channels_list=[config.feat_hidden_units, embedding_size], 
        item_t_in_channels=t_feat.size(-1), 
        item_t_hidden_channels_list=[config.feat_hidden_units, embedding_size], 

        bn=config.bn,
        num_clusters=config.num_clusters,
        num_samples=config.num_samples
    ).to(device)


use_clip_loss = False
use_mm_mf_loss = False


# return_all为True时：mig；为False时：mig_gt
def forward(g, return_all=False):
    if return_all:
        virtual_h, emb_h, t_h, v_h, encoded_t, encoded_v, z_memory_h = model(g, user_embeddings, v_feat, t_feat, 
                                                                 item_embeddings=item_embeddings if config.use_item_emb else None, 
                                                                 return_all=return_all)
    else:
        virtual_h = model(g, user_embeddings, v_feat, t_feat, item_embeddings=item_embeddings if config.use_item_emb else None, 
                          return_all=return_all)
    user_h = virtual_h[:num_users]
    item_h = virtual_h[num_users:]

    if return_all:

        user_emb_h = emb_h[:num_users]
        item_emb_h = emb_h[num_users:]

        user_t_h = t_h[:num_users]
        item_t_h = t_h[num_users:]

        if v_h is not None:
            user_v_h = v_h[:num_users]
            item_v_h = v_h[num_users:]
        else:
            user_v_h = None
            item_v_h = None

        return user_h, item_h, user_emb_h, item_emb_h, user_t_h, item_t_h, user_v_h, item_v_h, encoded_t, encoded_v, z_memory_h
    else:
        return user_h, item_h
    

        


# 计算评估指标
def evaluate(user_items_dict, mask_user_items_dict):
    # 模型切换评估模式：关闭droupout等训练专用层
    model.eval()
    # 前向传播获取最终嵌入
    user_h, item_h = forward(g)
    user_h = user_h.detach().cpu().numpy()
    item_h = item_h.detach().cpu().numpy()

    # 计算平均评估指标，并返回
    mean_results_dict = evaluate_mean_global_metrics(user_items_dict, mask_user_items_dict,
                                                    user_h, item_h, k_list=[10, 20], metrics=["precision","recall", "ndcg"])
    return mean_results_dict

# 更新学习率
def update_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        # 计算新学习率
        new_lr = param_group['lr'] * config.lr_decay
        # 新学习率不能低于最小值
        if new_lr >= config.lr_decay_min:
            param_group['lr'] = new_lr
#         param_group['lr'] = param_group['lr'] * lr_decay


# 用于训练的用户-物品交互边的数据加载器
train_edges_data_loader = create_tensors_dataloader(
        torch.arange(len(train_user_item_edges)),
        torch.tensor(train_user_item_edges), batch_size=config.batch_size, shuffle=True
)

# 动态调整学习率
optimizer = torch.optim.Adam([user_embeddings, item_embeddings] + list(model.parameters()), lr=config.lr)

# early_stop_metric = "ndcg@20"
early_stop_metric = "recall@20"
best_valid_score = 0.0 
early_stop_valid_results_dict = None
early_stop_test_results_dict = None
best_epoch = None

# 将两种来源的配置参数合并到一个字典中：命令行参数args、配置文件参数config
combined_config_dict = vars(args)
for k, v in asdict(config).items():
    combined_config_dict[k] = v


patience_count = 0
total_train_time = 0.0

# 日志目录
run_log_dir = "run_logs"
# 日志文件名
run_log_fname = "{}.json".format(args.dataset)
# 完整日志文件路径
run_log_fpath = os.path.join(run_log_dir, run_log_fname)


# NOTE 图神经网络推荐系统训练流程：
for epoch in range(1, config.num_epochs + 1):

    epoch_start_time = time.time()
    # 批处理数据加载
    for step, (batch_edge_indices, batch_edges) in enumerate(train_edges_data_loader):
        step_start_time = time.time()
        # 模型训练模式
        model.train()

      
        # 创建图的局部副本，避免污染原始图
        with g.local_scope():

            new_g = g

            # mig方法
            if method == "mig":
                user_h, item_h = forward(new_g)
            # mig_gt方法
            else:
                user_h, item_h, user_emb_h, item_emb_h, user_t_h, item_t_h, user_v_h, item_v_h, encoded_t, encoded_v, z_memory_h = forward(new_g, return_all=True)

            # 损失函数计算：BPR损失、L2正则化
            # infobpr = bpr by default
            mf_losses = compute_info_bpr_loss(user_h, item_h, batch_edges, num_negs=config.num_negs, reduction="none")
            l2_loss = compute_l2_loss([user_h, item_h])
            loss = mf_losses.sum() + l2_loss * config.l2_coef


            # mig_gt方法的多模态平滑损失
            if method != "mig":
                pos_user_h = user_h[batch_edges[:, 0]]
                pos_z_memory_h = z_memory_h[batch_edges[:, 1] + num_users]  
                unsmooth_logits = (pos_user_h.unsqueeze(1) @ pos_z_memory_h.permute(0, 2, 1)).squeeze(1)
                unsmooth_loss = F.cross_entropy(unsmooth_logits, torch.zeros([batch_edges.size(0)], dtype=torch.long, device=device), reduction="none").sum()
                loss = loss + unsmooth_loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_end_time = time.time()
    # 每轮epoch结束，更新学习率
    update_learning_rate(optimizer)

    epoch_end_time = time.time()
    total_train_time += epoch_end_time - epoch_start_time

  
    # NOTE 每轮epoch的日志
    print("epoch = {}\tloss = {:.4f}\tmf_loss = {:.4f}\tl2_loss = {:.4f}\tupdated_lr = {:.4f}\tepoch_time = {:.4f}s\tpcount = {}"
          .format(epoch, loss.item(), mf_losses.mean().item(), l2_loss.item(), optimizer.param_groups[0]['lr'], epoch_end_time-epoch_start_time, patience_count))
    
    # NOTE 早停机制
    if epoch % config.validation_freq == 0:
        print("\nEvaluation before epoch {} ......".format(epoch))

        valid_results_dict = evaluate(valid_user_items_dict, valid_mask_user_items_dict)
        print("valid_results_dict = ", valid_results_dict)


        current_score = valid_results_dict[early_stop_metric]
        # 性能提升
        if current_score > best_valid_score:

            test_results_dict = evaluate(test_user_items_dict, test_mask_user_items_dict)
            print("test_results_dict = ", test_results_dict)

            best_valid_score = current_score
            best_epoch = epoch
            early_stop_valid_results_dict = valid_results_dict
            early_stop_test_results_dict = test_results_dict

            print("updated early_stop_test_results_dict = ", early_stop_test_results_dict)
            patience_count = 0
        # 性能未提升
        else:
            print("old early_stop_test_results_dict = ", early_stop_test_results_dict)
            patience_count += config.validation_freq
            # 触发早停，config.patience是容忍的连续未提升轮数
            if patience_count >= config.patience:
                print("Early stopping at epoch {} ......".format(epoch))
                break
        


