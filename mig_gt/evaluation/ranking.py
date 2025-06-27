# coding=utf-8

from tqdm import tqdm
import numpy as np
import torch
from mig_gt.metrics.ranking import ndcg_score, precision_score, recall_score

from mig_gt.vector_search.vector_search import VectorSearchEngine


# 计算推荐系统的评估指标
def score(ground_truth, pred_items, k_list, metrics):
    # 预测正确为1，错误为0
    pred_match = [1 if item in ground_truth else 0 for item in pred_items]

    max_k = k_list[-1]
    if len(ground_truth) > max_k:
        ndcg_gold = [1] * max_k
    else:
        ndcg_gold = [1] * len(ground_truth) + [0] * (max_k - len(ground_truth))

    res_score = []
    # 使用 ndcg、precision、recall 等指标计算函数
    for metric in metrics:
        if metric == "ndcg":
            score_func = ndcg_score
        elif metric == "precision":
            score_func = precision_score
        elif metric == "recall":
            score_func = recall_score
        else:
            raise Exception("Not Found Metric : {}".format(metric))

        for k in k_list:
            if metric == "ndcg":
                res_score.append(score_func(ndcg_gold[:k], pred_match[:k]))
            else:
                res_score.append(score_func(ground_truth, pred_match[:k]))

    return res_score

# 计算评估指标（NDCG、Precision、Recall）在Top-K推荐中的表现
def evaluate_mean_global_metrics(user_items_dict, user_mask_items_dict,
                                 user_embedding, item_embedding,
                                 k_list=[10, 20], metrics=["ndcg"]):

    # 创建根据内积来搜索的item向量数据库
    v_search = VectorSearchEngine(item_embedding)

    # 将user嵌入转为numpy数组
    if isinstance(user_embedding, torch.Tensor):
        user_embedding = user_embedding.detach().cpu().numpy()
    else:
        user_embedding = np.asarray(user_embedding)
    # 待评估的user id列表
    user_indices = list(user_items_dict.keys())
    # 待评估的user id列表 的嵌入表示
    embedded_users = user_embedding[user_indices]
    # 计算所有用户需要屏蔽的物品数量的最大值
    # user_mask_items_dict是需要屏蔽的物品（eg：训练集中已）
    max_mask_items_length = max(len(user_mask_items_dict[user]) for user in user_indices)

    # 将user传入item向量数据库，通过内积的方式搜索出与这个user关联的item，一共k_list[-1] + max_mask_items_length个（确保过滤后仍有足够候选）
    _, user_rank_pred_items = v_search.search(embedded_users, k_list[-1] + max_mask_items_length)

    res_scores = []
    for user, pred_items in tqdm(zip(user_indices, user_rank_pred_items)):
        # user真实交互的item
        items = user_items_dict[user]
        # 需要屏蔽的item
        mask_items = user_mask_items_dict[user]
        # 过滤后取Top-K
        pred_items = [item for item in pred_items if item not in mask_items][:k_list[-1]]
        # 计算指标
        res_score = score(items, pred_items, k_list, metrics)

        res_scores.append(res_score)

    res_scores = np.asarray(res_scores)
    names = []
    # 将所有用户指标按 metric@k 格式命名
    for metric in metrics:
        for k in k_list:
            names.append("{}@{}".format(metric, k))

    # 返回各指标的平均值
    return dict(zip(names, np.mean(res_scores, axis=0, keepdims=False)))

