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

        self.softplus = nn.Softplus()
        self.flatten =  nn.Flatten()
        self.tanh = nn.Tanh()

        self.lstm_p = Recurrent(
            layer_num=1,
            state_shape=input_dim,
            action_shape=feature_dim,
            device=self.device,
            ).to(device) 
        self.fc1_p = MLP(input_dim=feature_dim,output_dim=1024,hidden_sizes=hidden_dim).to(device) 
        self.fc2_p = MLP(input_dim=1024,output_dim=feature_dim,hidden_sizes=hidden_dim).to(device) 

        self.lstm_t = Recurrent(
            layer_num=1,
            state_shape=input_dim,
            action_shape=feature_dim,
            device=self.device,
            ).to(device) 
        self.fc1_t = MLP(input_dim=feature_dim,output_dim=1024,hidden_sizes=hidden_dim).to(device) 
        self.fc2_t = MLP(input_dim=1024,output_dim=feature_dim,hidden_sizes=hidden_dim).to(device) 

        self.lstm_n = Recurrent(
            layer_num=1,
            state_shape=input_dim,
            action_shape=feature_dim,
            device=self.device,
            ).to(device) 
        self.fc1_n = MLP(input_dim=feature_dim,output_dim=1024,hidden_sizes=[256,256]).to(device) 
        self.fc2_n = MLP(input_dim=1024,output_dim=feature_dim,hidden_sizes=[256,256]).to(device) 
        self.output_dim = 2 * feature_dim
        self.hidden_p = None
        self.hidden_t = None
        self.hidden_n = None

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
        # 根据 tag 选择对应的 MLP
        for i in range(obs.shape[0]):
            lstm_logits,hidden_p = self.lstm_p(obs,state)
            fc1_logits = self.fc1_p(lstm_logits)
            fc1_logits = self.tanh(self.softplus(fc1_logits))+fc1_logits
            output_p = self.tanh(self.softplus(self.fc2_p(fc1_logits)))
            if torch.equal(obs[i, 2:4], torch.tensor([1, 0], device=self.device)):
                # print("track part acivated")
                lstm_logits,hidden_t = self.lstm_t(obs,state)
                fc1_logits = self.fc1_t(lstm_logits)
                fc1_logits = self.tanh(self.softplus(fc1_logits))+fc1_logits
                output_t = self.tanh(self.softplus(self.fc2_t(fc1_logits)))
                output = torch.cat((output_t, output_p),dim=1)
            elif torch.equal(obs[i, 2:4], torch.tensor([0, 1], device=self.device)):
                # print("safe part acivated")
                lstm_logits,hidden_n = self.lstm_n(obs,state)
                fc1_logits = self.fc1_n(lstm_logits)
                fc1_logits = self.tanh(self.softplus(fc1_logits))+fc1_logits
                output_n = self.tanh(self.softplus(self.fc2_n(fc1_logits)))
                output=torch.cat((output_n, output_p),dim=1)
            else:
                raise ValueError(
                    "Invalid tag value. Must be 1, 2, or 3.It's ", obs[i, 2:4]
                )
        state = {"hidden_p":hidden_p,"hidden_t":hidden_t,"hidden_n":hidden_n}
        return output, state


def main():
    # 模型参数
    input_dim = 34
    hidden_dim = [64, 64]
    feature_dim = 256
    device = "cuda:1"

    # 创建模型
    model = STAR(input_dim, hidden_dim, feature_dim, device)

    # 示例输入
    x = torch.randn(2, input_dim).to(device)  # 4个样本
    x[:, 2:4] = torch.tensor([0, 1])

    # 进行前向传播
    output, _ = model(
        torch.tensor(x),
    )
    print("Output:", output.shape)


if __name__ == "__main__":
    main()
