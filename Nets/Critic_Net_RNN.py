from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from tianshou.utils.net.common import MLP,Recurrent
from tianshou.utils.net.discrete import NoisyLinear
from tianshou.utils.net.common import Recurrent_GRU as Recurrent
import os

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


class Critic_Preprocess_Net(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        input_dim: int,
        device: Union[str, int, torch.device] = "cpu",
        nums_lstm:int = 3,
        hidden_size:  Sequence[int] =[128],
        feature_dim: int =128,
    ) -> None:
        super().__init__()
        self.device=device
        # self.lstm=Recurrent(nums_lstm,
        #                     input_dim,
        #                     feature_dim,
        #                     device=device,
        #                     hidden_layer_size=hidden_size,
        #                     ).to(device)
        self.output_dim=feature_dim
        # self.mlp=MLP(
        #     input_dim,
        #     feature_dim,
        #     hidden_size,
        #     device=self.device,
        #     flatten_input=False,
        # ).to(self.device)
        self.rnn = Recurrent(            
            layer_num=1,
            state_shape=input_dim,
            action_shape=feature_dim,
            device=self.device,)        
        self.output_dim=feature_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        mlp_logits,hidden_state=self.rnn(obs,state)
        # print(obs.shape)
        # print(obs.shape)
        # rnn_logits,self.state=self.lstm(obs,state)
        return mlp_logits,hidden_state

if __name__ == "__main__":
    Net=Critic_Preprocess_Net(2,'cuda')
    print(Net)
    x = torch.randn(6,18)
    res,state=Net.forward(x)
    print(res.shape)