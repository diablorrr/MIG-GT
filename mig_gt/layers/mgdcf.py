# coding=utf-8

import torch
import torch.nn as nn
import dgl
import numpy as np
import dgl.function as fn






class MGDCF(nn.Module):

    CACHE_KEY = "mgdcf_weight"

    def __init__(self, k, alpha, beta, x_drop_rate, edge_drop_rate, z_drop_rate, *args, **kwargs):
        """
        :param k: number of iterations.
        :param alpha: a hyperparameter of MGDCF.
        :param beta: a hyperparameter of MGDCF.
        :param x_drop_rate: dropout rate of input embeddings.
        :param edge_drop_rate: dropout rate of edge weights.
        :param z_drop_rate: dropout rate of output embeddings.
        """

        super().__init__(*args, **kwargs)

        self.k = k
        self.alpha = alpha
        self.beta = beta
        # 计算归一化系数
        self.gamma = torch.tensor(MGDCF.compute_gamma(alpha, beta, k)).float()
        self.gammas =[torch.tensor(MGDCF.compute_gamma(alpha, beta, i)).float() for i in range(1, k+1)]

        self.x_drop_rate = x_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.z_drop_rate = z_drop_rate

        # droupout
        self.x_dropout = nn.Dropout(x_drop_rate)
        self.edge_dropout = nn.Dropout(edge_drop_rate)
        self.z_dropout = nn.Dropout(z_drop_rate)



    # 计算归一化系数
    @classmethod
    def compute_gamma(cls, alpha, beta, k):
        return np.power(beta, k) + alpha * np.sum([np.power(beta, i) for i in range(k)])

    # @classmethod
    # def build_homo_graph(cls, user_item_edges, num_users=None, num_items=None):

    #     user_index, item_index = user_item_edges.T

    #     if num_users is None:
    #         num_users = np.max(user_index) + 1

    #     if num_items is None:
    #         num_items = np.max(item_index) + 1

    #     num_homo_nodes = num_users + num_items
    #     homo_item_index = item_index + num_users
    #     src = user_index
    #     dst = homo_item_index

    #     g = dgl.graph((src, dst), num_nodes=num_homo_nodes)
    #     g =  dgl.add_reverse_edges(g)
    #     # Different from LightGCN, MGDCF considers self-loop
    #     g = dgl.add_self_loop(g)
    #     g = dgl.to_simple(g)
        
    #     return g

    # 生成同构图
    @classmethod
    def build_sorted_homo_graph(cls, user_item_edges, num_users=None, num_items=None):
        # 用户-物品边(行：用户、物品；列：交互数) 转置后 (行：用户；列：物品)
        # 因此用于快速分离 所有 用户ID 和 物品ID
        user_index, item_index = user_item_edges.T # .T是转置张量

        if num_users is None:
            num_users = np.max(user_index) + 1

        if num_items is None:
            num_items = np.max(item_index) + 1

        user_index = torch.tensor(user_index)
        item_index = torch.tensor(item_index)

        # 图的总节点数
        num_homo_nodes = num_users + num_items
        # 重新映射物品ID [1,2,3] -> [10,20,30] ；目的：用户和物品在图中视为同一类型节点，但ID不重叠
        homo_item_index = item_index + num_users
        # 上下代码对照看：正向边（user_index->homo_item_index）反向边（homo_item_index->user_index）自环边（num_homo_nodes->num_homo_nodes）
        # 双向边用于信息双向传播，自环边用于保留节点自身信息
        src = torch.concat([user_index, homo_item_index, torch.arange(num_homo_nodes)], dim=0)
        dst = torch.concat([homo_item_index, user_index, torch.arange(num_homo_nodes)], dim=0)

        g = dgl.graph((src, dst), num_nodes=num_homo_nodes)

        assert g.num_edges() == src.size(0)

        # g =  dgl.add_reverse_edges(g)
        # # Different from LightGCN, MGDCF considers self-loop
        # g = dgl.add_self_loop(g)
        # g = dgl.to_simple(g)
        
        return g

    # 对邻接矩阵进行归一化，避免度数高的节点主导消息传递
    @classmethod
    @torch.no_grad()
    def norm_adj(cls, g):

        CACHE_KEY = MGDCF.CACHE_KEY

        # if CACHE_KEY in g.edata:
        #     return

        
        degs = g.in_degrees()
        src_norm = degs.pow(-0.5)
        dst_norm = src_norm

        with g.local_scope():
            g.ndata["src_norm"] = src_norm
            g.ndata["dst_norm"] = dst_norm
            g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", CACHE_KEY))
            gcn_weight = g.edata[CACHE_KEY]

        g.edata[CACHE_KEY] = gcn_weight


    def forward(self, g, x, return_all=False):

        CACHE_KEY = MGDCF.CACHE_KEY
        # 归一化边权重
        MGDCF.norm_adj(g)

        edge_weight = g.edata[CACHE_KEY]
        dropped_edge_weight = self.edge_dropout(edge_weight)

        # print(edge_weight)
        # asdfadsf

        h0 = self.x_dropout(x) # 输入嵌入
        h = h0

        if return_all:
            h_list = []

        with g.local_scope():
            g.edata[CACHE_KEY] = dropped_edge_weight

            # 迭代扩散k此
            for _ in range(self.k):
                g.ndata["h"] = h

                # NOTE 消息传递
                g.update_all(fn.u_mul_e("h", CACHE_KEY, "m"), fn.sum("m", "h"))
                h = g.ndata.pop("h")

                # 加权更新
                h = h * self.beta + h0 * self.alpha

                if return_all:
                    h_list.append(h)

        if not return_all:
            h = h / self.gamma
            h = self.z_dropout(h)
            return h
        else:                
            h_list = [h / gamma for h, gamma in zip(h_list, self.gammas)]
            h_list = [self.z_dropout(h) for h in h_list]
            return h_list

        # if return_all:
        #     return h_list[-1], h_list
        # else:
        #     return h_list[-1]
       
