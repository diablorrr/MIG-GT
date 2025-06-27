# coding = utf8

import torch
import numpy as np
import faiss


class VectorSearchEngine(object):
    def __init__(self, vectors):
        super().__init__()
        # 输入向量统一转为numpy数组
        if isinstance(vectors, torch.Tensor):
            self.vectors = vectors.detach().cpu().numpy()
        else:
            self.vectors = np.array(vectors)
        # 获取向量维度
        self.dim = self.vectors.shape[1]
        # 生成以内积方式搜索的向量数据库，且可存放的向量维度为dim
        self.index = faiss.IndexFlatIP(self.dim)
        # 将向量加入向量数据库
        self.index.add(self.vectors)

    def search(self, query_vectors, k=10):
        # 将输入list转换为numpy数组
        query_vectors = np.asarray(query_vectors)
        # 将查询向量query_vectors与向量数据库中的内容内积，返回里面相似度最高的k个值
        topK_distances, topK_indices = self.index.search(query_vectors, k)

        return topK_distances, topK_indices
