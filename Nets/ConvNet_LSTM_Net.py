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
        c: int,
        h: int,
        w: int,
        device: Union[str, int, torch.device] = "cpu",
        feature_dim: int = 128
    ) -> None:
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.output_dim = feature_dim
        self.convNet1 =nn.Conv2d(c, 16, kernel_size=8, stride=4)
        self.convNet2 =nn.Conv2d(16, 64, kernel_size=4, stride=2)
        self.relu =  nn.ReLU()
        self.flatten =  nn.Flatten()
        with torch.no_grad():
            input=torch.zeros(1, c, h, w)
            self.convd_output_dim = np.prod(self.convNet2(self.convNet1(input)).shape[1:])
        self.fc1 = MLP(input_dim=self.convd_output_dim,output_dim=256,hidden_sizes=[256,256])
        self.lstm = Recurrent(
            layer_num=1,
            state_shape=256,
            action_shape=feature_dim,
            device=self.device,)

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
        conv1_logits = self.convNet1(obs)
        conv2_logits = self.convNet2(conv1_logits)
        fc_logits = self.fc1(conv2_logits)
        logits, state = self.lstm(fc_logits,state)
        torch.cuda.empty_cache()
        return logits, state

if __name__ == "__main__":
    Net=Track_Net(3,250,250,'cuda',feature_dim=512).cuda()
    print(Net)
    x = torch.randn(1,3,250,250)
    res,state=Net.forward(x)
    print(res.shape)