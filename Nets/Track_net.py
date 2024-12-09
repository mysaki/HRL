from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from tianshou.utils.net.common import MLP
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
        input_dim: int,

        action_shape: Sequence[int],
        concat: bool = False,
        num_atoms: int = 1,
        softmax: bool = False,
        device: Union[str, int, torch.device] = "cpu",
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
        get_feature:bool = False,
        use_dueling: Optional[int] = False,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        feature_dim: int =128
    ) -> None:
        super().__init__()
        self.device = device
        self.get_feature = get_feature
        self.feature_dim = feature_dim
        self.use_dueling =use_dueling
        self.softmax=softmax
        self.num_atoms=num_atoms
        self.model = MLP(
            input_dim,  # type: ignore
            feature_dim,
            [feature_dim],
            device=self.device
        )

        self.output_dim=feature_dim

        if use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_shape, num_atoms
            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": self.feature_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": self.feature_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.model(obs)
        bsz = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state

if __name__ == "__main__":
    Net=Track_Net(3,250,250,2,'cuda',output_dim=512,layer_init=layer_init).cuda()
    print(Net)
    x = torch.randn(1,3,250,250)
    res,state=Net.forward(x)
    print(res.shape)