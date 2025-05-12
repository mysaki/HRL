import torch
import torch.nn as nn
from tianshou.utils.net.common import MLP
# from tianshou.utils.net.common import Recurrent
from tianshou.utils.net.common import Recurrent_GRU as Recurrent
import copy
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
import os
from .KURL_Net import Track_Net as KURL
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class STAR(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        feature_dim,
        device,
        norm_layers=None,
        laser_dim=32,
        dropout=0.1,
    ):
        super(STAR, self).__init__()
        self.device = device
        self.laser_dim = laser_dim
        self.soft_max = nn.Softmax(dim=2)

        # MLP层
        self.mlp_t = KURL(
        input_dim=input_dim, device=self.device, feature_dim=feature_dim).to(self.device)
        self.mlp_n = KURL(
        input_dim=input_dim, device=self.device, feature_dim=feature_dim).to(self.device)
        self.mlp_p = KURL(
        input_dim=input_dim, device=self.device, feature_dim=feature_dim).to(self.device)
        self.output_dim = 2 * feature_dim
        self.hidden_p = None
        self.hidden_b = None
        self.hidden_c = None

    def fuse_networks(self, model_a, model_b):
        # 确保模型结构相同
        assert isinstance(model_a, MLP) and isinstance(
            model_b, MLP
        ), "Both models must be MLP instances."

        # 初始化融合模型
        fused_model = copy.deepcopy(model_a)

        # 融合 A 和 B 网络的权重和偏差
        for a_param, b_param, fused_param in zip(
            model_a.parameters(), model_b.parameters(), fused_model.parameters()
        ):
            if len(a_param.shape) == 2:  # 权重
                fused_param.data = a_param.data * b_param.data
            elif len(a_param.shape) == 1:  # 偏差
                fused_param.data = a_param.data + b_param.data

        return fused_model

    def forward(self, obs, state=None, info={}):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if state != None:
            hidden_p = state["hidden_p"]
            hidden_t = state["hidden_t"]
            hidden_n = state["hidden_n"]
        else:
            hidden_p = None
            hidden_t = None
            hidden_n = None          
        output = []

        # 根据 tag 选择对应的 MLP
        for i in range(obs.shape[0]):
            output_p,hidden_p = self.mlp_p(obs[i].unsqueeze(dim=0),hidden_p)
            if torch.equal(obs[i, 2:4], torch.tensor([1, 0], device=self.device)):
                output_t,hidden_t = self.mlp_t(obs[i].unsqueeze(dim=0),hidden_t)
                output.append(torch.cat((output_t, output_p)))
            elif torch.equal(obs[i, 2:4], torch.tensor([0, 1], device=self.device)):
                output_n,hidden_n = self.mlp_n(obs[i].unsqueeze(dim=0),hidden_n)
                output.append(torch.cat((output_n, output_p)))
            else:
                raise ValueError(
                    "Invalid tag value. Must be 1, 2, or 3.It's ", obs[i, 2:4]
                )
        state = {"hidden_p":hidden_p,"hidden_t":hidden_t,"hidden_n":hidden_n}
        return torch.stack(output), state


def main():
    # 模型参数
    input_dim = 36
    hidden_dim = 512
    feature_dim = 128
    device = "cuda:1"

    # 创建模型
    model = STAR(input_dim, hidden_dim, feature_dim, device)

    # 示例输入
    x = torch.randn(4, input_dim).to(device)  # 4个样本
    x[:, 2:4] = torch.tensor([0, 1])

    # 进行前向传播
    output, _ = model(
        x,
    )
    print("Output:", output.shape)


if __name__ == "__main__":
    main()
