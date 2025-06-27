import random
import numpy as np
import torch
import dgl




def reset_seed(seed):
    print("set seed: {} ...".format(seed))
    # python 内置的随机种子
    random.seed(seed)
    # NumPy 随机种子
    np.random.seed(seed)

   

    # PyTorch CPU 的随机种子
    torch.manual_seed(seed)
    # PyTorch GPU 的随机种子
    torch.cuda.manual_seed_all(seed)
    # 确保 CuDNN 的卷积计算是确定的（避免 GPU 计算的随机性）
    torch.backends.cudnn.deterministic = True
    # DGL（Deep Graph Library） 的随机种子
    dgl.seed(seed)



