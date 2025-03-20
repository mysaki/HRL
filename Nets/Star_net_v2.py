import torch
import torch.nn as nn
from tianshou.utils.net.common import MLP
# from tianshou.utils.net.common import Recurrent
from tianshou.utils.net.common import Recurrent_GRU as Recurrent
import copy
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention

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

        # 循环神经网络层
        # self.rnn_p = Recurrent(
        #     layer_num=1,
        #     state_shape=laser_dim,
        #     action_shape=laser_dim,
        #     device=self.device,
        # ).to(self.device)
        # self.rnn_b = Recurrent(
        #     layer_num=1,
        #     state_shape=laser_dim,
        #     action_shape=laser_dim,
        #     device=self.device,
        # ).to(self.device)
        # self.rnn_c = Recurrent(
        #     layer_num=1,
        #     state_shape=laser_dim,
        #     action_shape=laser_dim,
        #     device=self.device,

        # ).to(self.device)
        # # 注意力层
        # self.sa_p = ScaledDotProductAttention(
        #     d_model=laser_dim, d_k=laser_dim, d_v=laser_dim, h=8
        # ).to(self.device)
        # self.sa_b = ScaledDotProductAttention(
        #     d_model=laser_dim, d_k=laser_dim, d_v=laser_dim, h=8
        # ).to(self.device)
        # self.sa_c = ScaledDotProductAttention(
        #     d_model=laser_dim, d_k=laser_dim, d_v=laser_dim, h=8
        # ).to(self.device)

        # 归一化层+Dropout层
        # self.norm_p = nn.LayerNorm(laser_dim).to(self.device)
        # self.dropout_p = nn.Dropout(dropout).to(self.device)
        # self.norm_b = nn.LayerNorm(laser_dim).to(self.device)
        # self.dropout_b = nn.Dropout(dropout).to(self.device)
        # self.norm_c = nn.LayerNorm(laser_dim).to(self.device)
        # self.dropout_c = nn.Dropout(dropout).to(self.device)

        self.hidden_p = None
        self.hidden_b = None
        self.hidden_c = None

        # MLP层
        self.mlp_a = MLP(
            input_dim,
            feature_dim,
            hidden_dim,
            device=device,
            norm_layer=norm_layers,
            flatten_input=False,
        ).to(self.device)
        self.mlp_b = MLP(
            input_dim,
            feature_dim,
            hidden_dim,
            device=device,
            norm_layer=norm_layers,
            flatten_input=False,
        ).to(self.device)
        self.mlp_c = MLP(
            input_dim,
            feature_dim,
            hidden_dim,
            device=device,
            norm_layer=norm_layers,
            flatten_input=False,
        ).to(self.device)
        self.mlp_public = MLP(
            input_dim,
            feature_dim,
            hidden_dim,
            norm_layer=norm_layers,
            device=device,
            flatten_input=False,
        ).to(self.device)
        self.output_dim = feature_dim
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
            hidden_b = state["hidden_b"]
            hidden_c = state["hidden_c"]
        else:
            hidden_p = None
            hidden_b = None
            hidden_c = None          
        # fuse_a = self.fuse_networks(self.mlp_a,self.mlp_public)
        self.fuse_b = self.fuse_networks(self.mlp_b,self.mlp_public)
        self.fuse_c = self.fuse_networks(self.mlp_c, self.mlp_public)

        # print("star:",obs.shape)
        # output_b = self.mlp_b(obs)
        output = []
        for i in range(obs.shape[0]):
            if torch.equal(obs[i, 2:4], torch.tensor([1, 0], device=self.device)):
                output_b = self.fuse_b(obs[i])
                output.append(output_b)
            elif torch.equal(obs[i, 2:4], torch.tensor([0, 1], device=self.device)):
                output_c = self.fuse_c(obs[i])
                output.append(output_c)

        # # 根据 tag 选择对应的 MLP
        # for i in range(obs.shape[0]):
        #     # if torch.equal(obs[i,2:4], torch.tensor([0,0],device=self.device)):
        #     #     output.append(fuse_a(obs[i]))
        #     # laser_data = obs[i, -self.laser_dim: ]
        #     # laser_rnn, hidden_p = self.rnn_p(
        #     #     laser_data.unsqueeze(dim=0), hidden_p
        #     # )
        #     # laser_rnn = laser_rnn.unsqueeze(dim=0)
        #     # laser_att = self.norm_p(self.dropout_p(laser_att) + laser_rnn)
        #     # laser_att = self.soft_max(laser_data)
        #     # combined_data = torch.cat((laser_att[0][0], obs[i]), dim=-1)
        #     output_p = self.mlp_public(obs[i])
        #     if torch.equal(obs[i, 2:4], torch.tensor([1, 0], device=self.device)):
        #         # print("track part acivated")
        #         # laser_data = obs[i, -self.laser_dim: ]
        #         # laser_rnn, hidden_b = self.rnn_b(
        #         #     laser_data.unsqueeze(dim=0), hidden_b
        #         # )
        #         # laser_rnn = laser_rnn.unsqueeze(dim=0)
        #         # laser_att = self.soft_max(laser_data)
        #         # laser_att = self.norm_b(self.dropout_b(laser_att) + laser_rnn)
        #         # combined_data = torch.cat((laser_att[0][0], obs[i]), dim=-1)
        #         output_b = self.mlp_b(obs[i])
        #         output.append(torch.cat((output_b, output_p)))
        #     elif torch.equal(obs[i, 2:4], torch.tensor([0, 1], device=self.device)):
        #         # print("safe part acivated")
        #         # laser_data = obs[i, -self.laser_dim: ]
        #         # laser_rnn, hidden_c = self.rnn_c(
        #         #     laser_data.unsqueeze(dim=0), hidden_c
        #         # )
        #         # laser_rnn = laser_rnn.unsqueeze(dim=0)
        #         # laser_att = self.soft_max(laser_data)
        #         # laser_att = self.norm_c(self.dropout_c(laser_att) + laser_rnn)
        #         # combined_data = torch.cat((laser_att[0][0], obs[i]), dim=-1)
        #         output_c = self.mlp_c(obs[i])
        #         output.append(torch.cat((output_c, output_p)))
        #     else:
        #         raise ValueError(
        #             "Invalid tag value. Must be 1, 2, or 3.It's ", obs[i, 2:4]
        #         )
        # state = {"hidden_p":hidden_p,"hidden_b":hidden_b,"hidden_c":hidden_c}
        return torch.stack(output), state


def main():
    # 模型参数
    input_dim = 36
    hidden_dim = [64, 64]
    feature_dim = 128
    device = "cuda:1"

    # 创建模型
    model = STAR(input_dim, hidden_dim, feature_dim, device)

    # 示例输入
    x = torch.randn(4, input_dim).to(device)  # 4个样本
    x[:, 2:4] = torch.tensor([0, 1])

    # 进行前向传播
    output, _ = model(
        torch.tensor(x),
    )
    print("Output:", output.shape)


if __name__ == "__main__":
    main()
