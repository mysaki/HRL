import torch
import torch.nn as nn
from tianshou.utils.net.common import MLP
# from tianshou.utils.net.common import Recurrent
from tianshou.utils.net.common import Recurrent_GRU as Recurrent
import copy
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention

class Net(nn.Module):

    def __init__(
        self,
        input_dim: int = 34,
        feature_dim: int = 256,
        device: str = 'cpu',
        norm_layers=None,
        dropout=0.1,
        action_dim = 2*51,
        hidden_size = [128,128]
    ):
        super(Net, self).__init__()
        self.device = device
        self.soft_max = nn.Softmax(dim=2)

        # 循环神经网络层
        self.rnn = Recurrent(
            layer_num=1,
            state_shape=input_dim,
            action_shape=feature_dim,
            device=self.device,
        ).to(self.device)
        self.fc = MLP(input_dim=feature_dim,output_dim=action_dim,hidden_sizes=hidden_size,device=self.device).to(self.device)

    def forward(self, obs, state=None, info={}):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits,state = self.rnn(obs,state)
        logits = self.fc(logits)
        bsz = logits.shape[0]
        logits = logits.view(bsz, -1, 51)
        return logits, state


if __name__ == "__main__":
    net = Net(input_dim = 34 ,feature_dim = 512,device = 'cuda:1')
    x = torch.randn(3,34).to('cuda:1')
    logits,state = net(x)
    bsz = logits.shape[0]
    logits = logits.view(bsz, -1, 51)
    print(logits.shape)

