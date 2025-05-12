import torch
import config_dqn as config
from Nets.Star_net import STAR
import numpy as np
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.common import ActorCritic, Net
from torch.distributions import Distribution, Independent, Normal
from tianshou.policy import PPOPolicy
import os
def get_low_policy(env=None):
    low_policy_state_dim = list(config.state_shape)
    low_policy_state_dim[-1]+=2
    low_policy_state_dim = tuple(low_policy_state_dim)
    low_policy_APNet = STAR(
        input_dim=np.prod(low_policy_state_dim), device=config.device, feature_dim=256, hidden_dim=[128,128])
    # low_policy_CPNet=Critic_Preprocess_Net(input_dim=np.prod(low_policy_state_dim), 
    #                             device=config.device,
    #                             feature_dim=256, 
    #                             hidden_size=[128,128]
    #                             ).to(config.device)
    low_policy_CPNet = STAR(input_dim=np.prod(low_policy_state_dim), feature_dim=256, device=config.device,hidden_dim=[128,128])
    low_policy_actor = ActorProb(low_policy_APNet,
                    config.action_shape,
                    unbounded=True,
                    hidden_sizes=[128],
                    device=config.device
                    ).to(config.device)
    low_policy_critic = Critic(
        low_policy_CPNet,
        hidden_sizes=[128],
        device=config.device,
    ).to(config.device)
    low_policy_actor_critic = ActorCritic(low_policy_actor, low_policy_critic)
    # orthogonal initialization
    for m in low_policy_actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    for parameter in low_policy_actor_critic.parameters():
        parameter.requires_grad = False
    low_level_optim = torch.optim.Adam(low_policy_actor_critic.parameters(), lr=config.lr)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(loc_scale) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)
    low_level_policy = PPOPolicy(
        actor=low_policy_actor,
        critic=low_policy_critic,
        optim=low_level_optim,
        dist_fn=dist,
        discount_factor=config.gamma,
        max_grad_norm=config.low_policy_params["max_grad_norm"],
        eps_clip=config.low_policy_params["eps_clip"],
        vf_coef=config.low_policy_params["vf_coef"],
        ent_coef=config.low_policy_params["ent_coef"],
        reward_normalization=config.low_policy_params["reward_normalization"],
        advantage_normalization=config.low_policy_params["advantage_normalization"],
        recompute_advantage=config.low_policy_params["recompute_advantage"],
        dual_clip=config.low_policy_params["dual_clip"],
        value_clip=config.low_policy_params["value_clip"],
        gae_lambda=config.low_policy_params["gae_lambda"],
        action_space=config.low_action_space,
        action_bound_method='clip',
    )
    log_low_model_path = "Log/join_train_track_ppo_1_18_10_29"

    print(f"Loading low level agent under {log_low_model_path}")
    ckpt_path = os.path.join(log_low_model_path, "Track_test.pth")
    if os.path.exists(ckpt_path):
        # checkpoint = torch.load(ckpt_path, map_location=args.device)
        # policy.load_state_dict(checkpoint["model"])
        # optim.load_state_dict(checkpoint["optim"])
        low_level_policy.load_state_dict(torch.load(ckpt_path))
        low_level_policy.to(config.device)
        low_level_policy.eval()
        print('Low level policy load!')
    else:
        print("Fail to restore low level policy.")
    return low_level_policy