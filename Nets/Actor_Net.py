# from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union
# import numpy as np
# import torch
# from torch import nn
# from tianshou.utils.net.common import MLP,Recurrent
# from tianshou.utils.net.discrete import NoisyLinear


# def layer_init(
#     layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
# ) -> nn.Module:
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer


# def scale_obs(module: Type[nn.Module], denom: float = 255.0) -> Type[nn.Module]:

#     class scaled_module(module):
#         def forward(
#             self,
#             obs: Union[np.ndarray, torch.Tensor],
#             state: Optional[Any] = None,
#             info: Dict[str, Any] = {}
#         ) -> Tuple[torch.Tensor, Any]:
#             return super().forward(obs / denom, state, info)

#     return scaled_module


# class Actor_Preprocess_Net(nn.Module):
#     """Reference: Human-level control through deep reinforcement learning.

#     For advanced usage (how to customize the network), please refer to
#     :ref:`build_the_network`.
#     """

#     def __init__(
#         self,
#         image_dim: list,
#         device: Union[str, int, torch.device] = "cpu",
#         hidden_size: int=128,
#         nums_lstm:int = 3,
#         feature_dim: int =128,
#     ) -> None:
#         super().__init__()
#         self.device = device
#         self.feature_dim = feature_dim
#         c,h,w=image_dim # 1,256,256
#         self.conv_net = nn.Sequential(
#             layer_init(nn.Conv2d(c, 128, kernel_size=8, stride=4)),
#             nn.ReLU(inplace=True),
#             layer_init(nn.Conv2d(128, 64, kernel_size=4, stride=2)),
#             nn.ReLU(inplace=True),
#             layer_init(nn.Conv2d(64, 32, kernel_size=3, stride=1)),
#             nn.ReLU(inplace=True),
#             nn.Flatten()
#         ).to(self.device)
#         with torch.no_grad():
#             input=torch.zeros(1, c, h, w).to(self.device)
#             self.convd_output_dim = np.prod(self.conv_net(input).shape[1:])

#         self.lstm=Recurrent(nums_lstm,
#                             self.convd_output_dim,
#                             feature_dim,
#                             device=self.device,
#                             hidden_layer_size=hidden_size,
#                             ).to(self.device)
#         self.mlp=MLP(
#             self.convd_output_dim,
#             feature_dim,
#             [128],
#             device=self.device,
#             flatten_input=False,
#         ).to(self.device)
#         self.output_dim=feature_dim+16

#     def forward(
#         self,
#         obs: Union[np.ndarray, torch.Tensor],
#         state: Optional[Any] = None,
#         info: Dict[str, Any] = {},
#     ) -> Tuple[torch.Tensor, Any]:
#         r"""Mapping: s -> Q(s, \*)."""
#         obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
#         laser_data=obs[:,2:18]
#         img_obs=obs[:,18:]
#         # print(obs.shape)
#         img_obs=img_obs.reshape([img_obs.shape[0],1,256,256])
#         # print(obs.shape)
#         # if len(obs.shape) == 3:
#         #     obs = np.array([obs])
#         logits = self.conv_net(img_obs)
        
#         # print("logits:",logits.shape)
#         # cat_logits=torch.cat((logits,laser_data),1)
#         # print("cat_logits:",cat_logits.shape)
#         mlp_logits=self.mlp(logits)
#         obs=torch.cat((obs,mlp_logits),1)
#         cat_logits=torch.cat((mlp_logits,laser_data),1)
#         # rnn_logits,self.state=self.lstm(logits,state)
#         # logits = self.model(obs)
#         # return logits, state
#         return cat_logits,state,obs

# if __name__ == "__main__":
#     Net=Actor_Preprocess_Net([1,256,256],'cuda')
#     # print(Net)
#     state=torch.randn(3,65554).to('cuda')
#     res,state=Net.forward(state)
#     print(res.shape)
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


class Actor_Preprocess_Net(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        input_dim: int,
        device: Union[str, int, torch.device] = "cpu",
        nums_lstm:int = 3,
        hidden_size: Sequence[int] =[128],
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
        self.mlp=MLP(
            input_dim,
            feature_dim,
            hidden_size,
            device=self.device,
            flatten_input=False,
        ).to(self.device)
        self.output_dim=feature_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        mlp_logits=self.mlp(obs)
        # print(obs.shape)
        # print(obs.shape)
        # rnn_logits,self.state=self.lstm(obs,state)
        return mlp_logits,state

if __name__ == "__main__":
    Net=Actor_Preprocess_Net(18,'cuda')
    print(Net)
    x = torch.randn(6,18)
    res,state=Net.forward(x)
    print(res.shape)