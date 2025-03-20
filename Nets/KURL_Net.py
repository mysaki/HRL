from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from tianshou.utils.net.common import MLP,Recurrent
from tianshou.utils.net.discrete import NoisyLinear


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def scale_obs(module: Type[nn.Module], denom: float = 255.0) -> Type[nn.Module]:

    class scaled_module(module):
        def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {}
        ) -> Tuple[torch.Tensor, Any]:
            return super().forward(obs / denom, state, info)

    return scaled_module


class Track_Net(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        input_dim: int = 2,
        device: Union[str, int, torch.device] = "cpu",
        feature_dim: int = 512
    ) -> None:
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.output_dim = feature_dim
        self.relu =  nn.ReLU()
        self.softplus = nn.Softplus()
        self.flatten =  nn.Flatten()
        self.tanh = nn.Tanh()
        self.lstm = Recurrent(
            layer_num=1,
            state_shape=input_dim,
            action_shape=feature_dim,
            device=self.device,
            )
        self.fc1 = MLP(input_dim=512,output_dim=1024,hidden_sizes=[256,256])
        self.fc2 = MLP(input_dim=1024,output_dim=feature_dim,hidden_sizes=[256,256])


    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        if len(obs.shape) == 3:
            obs = np.array([obs])
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        lstm_logits,state = self.lstm(obs,state)
        fc1_logits = self.fc1(lstm_logits)
        fc1_logits = self.tanh(self.softplus(fc1_logits))+fc1_logits
        fc2_logits = self.tanh(self.softplus(self.fc2(fc1_logits)))

        torch.cuda.empty_cache()
        return fc2_logits, state

if __name__ == "__main__":
    Net=Track_Net(2,'cuda',feature_dim=512).cuda()
    print(Net)
    x = torch.randn(3,2)
    res,state=Net.forward(x)
    print(res.shape)