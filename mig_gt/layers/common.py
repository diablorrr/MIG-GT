from torch import nn
import torch
import torch.nn.functional as F

# 自定义线性层
class MyLinear(nn.Linear):
    # pass
    # 重写 nn.Linear 的 reset_parameters方法
    def reset_parameters(self) -> None:
        # 权重初始化方法从默认的 Kaiming -> xavier_uniform_(均匀分布)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)



# 自定义线性层
class MyNormalLinear(nn.Linear):
    # pass
    # 重写 nn.Linear 的 reset_parameters方法
    def reset_parameters(self) -> None:
        # 权重初始化方法从默认的 Kaiming -> xavier_uniform_(正态分布)
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


# 自定义激活函数
class MyPReLU(nn.Module):

    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()

        # use alpha instead of weight
        self.alpha = nn.parameter.Parameter(torch.empty(num_parameters, **factory_kwargs).fill_(init))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.prelu(input, self.alpha)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


# 创建神经网络中的自定义操作
class Lambda(nn.Module):
    def __init__(self, func) -> None:
        super().__init__()

        self.func = func

    def forward(self, x):
        return self.func(x)


# 工厂函数：创建激活函数
def create_act(name=None):
    if name == "softmax":
        return nn.Softmax(dim=-1)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return Lambda(lambda x: x)
    else:
        raise Exception()


# 工厂函数：创建激活函数（支持激活函数的类型不同）
def get_activation(activation):
    if activation == "prelu":
        return MyPReLU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation is None or activation == "none":
        return torch.nn.Identity()
    else:
        raise NotImplementedError()
    


# 自定义多层感知机
class MyMLP(nn.Module):
    def __init__(self, in_channels, units_list, activation, drop_rate, bn, output_activation, output_drop_rate, output_bn, ln=False, output_ln=False):
        super().__init__()

        layers = []
        units_list = [in_channels] + units_list  # Add in_channels to the list of units

        for i in range(len(units_list) - 1):
            layers.append(MyLinear(units_list[i], units_list[i+1]))  # 添加线性层

            # 最后一层
            if i < len(units_list) - 2:
                if bn:
                    layers.append(nn.BatchNorm1d(units_list[i+1]))  # Add a batch normalization layer

                if ln:
                    layers.append(nn.LayerNorm(units_list[i+1]))
                
                layers.append(get_activation(activation))  # Add the PReLU activation function
                layers.append(nn.Dropout(drop_rate))
            # 非最后一层
            else:
                if output_bn:
                    layers.append(nn.BatchNorm1d(units_list[i+1]))

                if output_ln:
                    layers.append(nn.LayerNorm(units_list[i+1]))

                layers.append(get_activation(output_activation))  # Add the PReLU activation function
                layers.append(nn.Dropout(output_drop_rate))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
