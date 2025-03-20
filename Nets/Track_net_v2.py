<<<<<<< HEAD
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
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
        get_feature:bool = False,
        feature_dim: int =128
    ) -> None:
        super().__init__()
        self.device = device
        self.get_feature = get_feature
        self.feature_dim = feature_dim
        self.conv_net = nn.Sequential(
            layer_init(nn.Conv2d(c, 256, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(256, 128, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(128, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            input=torch.zeros(1, c, h, w)
            self.convd_output_dim = np.prod(self.conv_net(input).shape[1:])
        if self.get_feature == True:
            self.model=nn.Sequential(
            self.conv_net,
            layer_init(nn.Linear(self.convd_output_dim,feature_dim)),
            nn.ReLU(inplace=True),
            )
            self.output_dim=feature_dim
        else:

            self.model = nn.Sequential(
                self.conv_net,
                layer_init(nn.Linear(self.convd_output_dim,128)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(128,feature_dim)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(feature_dim,np.prod(action_shape))),
                nn.ReLU(inplace=True),
            )

            self.output_dim=np.prod(action_shape)

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
        return self.model(obs), state

if __name__ == "__main__":
    Net=Track_Net(3,250,250,2,'cuda',output_dim=512,layer_init=layer_init).cuda()
    print(Net)
    x = torch.randn(1,3,250,250)
    res,state=Net.forward(x)
=======
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
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
        get_feature:bool = False,
        feature_dim: int =128
    ) -> None:
        super().__init__()
        self.device = device
        self.get_feature = get_feature
        self.feature_dim = feature_dim
        self.conv_net = nn.Sequential(
            layer_init(nn.Conv2d(c, 256, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(256, 128, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(128, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            input=torch.zeros(1, c, h, w)
            self.convd_output_dim = np.prod(self.conv_net(input).shape[1:])
        if self.get_feature == True:
            self.model=nn.Sequential(
            self.conv_net,
            layer_init(nn.Linear(self.convd_output_dim,feature_dim)),
            nn.ReLU(inplace=True),
            )
            self.output_dim=feature_dim
        else:

            self.model = nn.Sequential(
                self.conv_net,
                layer_init(nn.Linear(self.convd_output_dim,128)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(128,feature_dim)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(feature_dim,np.prod(action_shape))),
                nn.ReLU(inplace=True),
            )

            self.output_dim=np.prod(action_shape)

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
        return self.model(obs), state

if __name__ == "__main__":
    Net=Track_Net(3,250,250,2,'cuda',output_dim=512,layer_init=layer_init).cuda()
    print(Net)
    x = torch.randn(1,3,250,250)
    res,state=Net.forward(x)
>>>>>>> f40c7b3 (代码上传)
    print(res.shape)