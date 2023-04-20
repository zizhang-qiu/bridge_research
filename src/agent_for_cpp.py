"""
@file:agent_for_cpp
@author:qzz
@date:2023/2/23
@encoding:utf-8
"""
from typing import Optional, Tuple

import torch
from torch import nn

from nets import PolicyNet, ValueNet


class SingleEnvAgent(nn.Module):
    """An agent implementation which can be used in c++ by torch.jit.script"""

    def __init__(self, p_net: PolicyNet, v_net: Optional[ValueNet] = None):
        super().__init__()
        self.p_net = p_net
        self.v_net = v_net if v_net is not None else ValueNet()

    @torch.jit.export
    def get_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get probabilities for given obs
        Args:
            obs (torch.Tensor): The obs tensor, should be 2-dimensional

        Returns:
            torch.Tensor: The probabilities.
        """
        state_tensor = obs[:, :480]
        legal_actions = obs[:, 480:518]
        # print("legal_actions", legal_actions)
        probs = torch.exp(self.p_net(state_tensor))
        # print("probs", probs)
        return probs * legal_actions

    @torch.jit.export
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Get an action for given obs
        Args:
            obs (torch.Tensor): The obs tensor, should be 1-dimensional

        Returns:
            torch.Tensor: The action

        """
        # single env obs is always 1d
        greedy = obs[-1].item()
        legal_actions = obs[480:518]
        obs_ = obs.unsqueeze(0)
        log_probs = self.get_log_probs(obs_).squeeze()
        probs = torch.exp(log_probs) * legal_actions
        available = True
        # print(probs)
        if greedy:
            action = torch.argmax(probs)
        else:
            if torch.equal(probs, torch.zeros_like(probs)):
                # print("Warning: all the probs are zero")
                action = torch.multinomial(legal_actions, 1)
                available = False
            else:
                action = torch.multinomial(probs, 1).squeeze()
        return action, log_probs.detach(), available

    @torch.jit.export
    def get_log_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities for given obs
        Args:
            obs (torch.Tensor): The obs tensor, should be 2-dimensional

        Returns:
            torch.Tensor: The log probabilities
        """
        state_tensor = obs[:, :480]
        # legal_actions = obs[:, 480:518]
        # print("legal_actions", legal_actions)
        probs = self.p_net(state_tensor)
        # print("probs", probs)
        return probs

    def compute_policy_gradient_loss(self, batch_obs: torch.Tensor, batch_action: torch.Tensor,
                                     batch_reward: torch.Tensor, batch_log_probs: torch.Tensor,
                                     clip_eps: float, entropy_ratio: float):
        """
        Compute policy gradient loss
        Args:
            batch_log_probs: The batch log probs tensor,shape (batch_size, num_actions)
            clip_eps: The epsilon in PPO paper
            batch_obs: The batch obs tensor, shape (batch_size, obs_size)
            batch_action: The batch action tensor, shape (batch_size,)
            batch_reward: The batch reward tensor, shape (batch_size,)
            entropy_ratio: The ratio of entropy, i.e. beta in A3C paper.

        Returns:
            The policy gradient loss, which is a scalar
        """
        current_log_probs = self.get_log_probs(batch_obs)
        current_action_log_probs = current_log_probs.gather(1, batch_action.long().unsqueeze(1)).squeeze(1)
        old_action_log_probs = batch_log_probs.gather(1, batch_action.long().unsqueeze(1)).squeeze(1)

        ratio = torch.exp(current_action_log_probs - old_action_log_probs)
        advantage = batch_reward - self.get_values(batch_obs).squeeze()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        surr1 = ratio * (advantage.detach())
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * (advantage.detach())
        current_probs = torch.exp(current_log_probs)
        entropy = -torch.sum(current_probs * current_log_probs, dim=-1)

        policy_loss = -torch.min(surr1, surr2) - entropy * entropy_ratio
        return policy_loss.mean()

    def get_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get values for given obs tensor
        Args:
            obs: The obs tensor, should be 2-dimensional

        Returns:
            The values given by v_net, 2-dimensional
        """
        state_tensor = obs[:, :480]
        values = self.v_net(state_tensor)
        return values

    def compute_a2c_loss(self, batch_obs: torch.Tensor, batch_action: torch.Tensor,
                         batch_reward: torch.Tensor, entropy_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantage actor critic loss
        Args:
            batch_obs: The batch obs tensor, shape (batch_size, obs_size)
            batch_action: The batch action tensor, shape (batch_size,)
            batch_reward: The batch reward tensor, shape (batch_size,)
            entropy_ratio: The ratio of entropy, i.e. beta in A3C paper.

        Returns:
            The A2C loss, which is a scalar
        """
        log_probs = self.get_log_probs(batch_obs)
        action_log_probs = log_probs.gather(1, batch_action.long().unsqueeze(1)).squeeze(1)
        advantage = batch_reward - self.get_values(batch_obs).squeeze()
        policy_loss = - (action_log_probs * advantage.detach())
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1) * entropy_ratio
        value_loss = torch.pow(advantage, 2)
        return (policy_loss - entropy).mean(), value_loss.mean()

    def compute_a2c_loss_with_clip(self, batch_obs: torch.Tensor, batch_action: torch.Tensor,
                                   batch_reward: torch.Tensor, batch_log_probs: torch.Tensor, entropy_ratio: float,
                                   clip_eps: float):
        """
        Compute A2C loss with importance ratio clipping
        Args:
            batch_obs: The batch obs tensor, shape (batch_size, obs_size)
            batch_action: The batch action tensor, shape (batch_size,)
            batch_reward: The batch reward tensor, shape (batch_size,)
            batch_log_probs: The batch log probs tensor, shape (batch_size, num_actions)
            entropy_ratio: The ratio of entropy, i.e. beta in A3C paper.
            clip_eps: The epsilon in PPO paper

        Returns:
            The A2C loss with importance ratio clipping
        """
        current_log_probs = self.get_log_probs(batch_obs)
        current_action_log_probs = current_log_probs.gather(1, batch_action.long().unsqueeze(1)).squeeze(1)
        old_action_log_probs = batch_log_probs.gather(1, batch_action.long().unsqueeze(1)).squeeze(1)

        ratio = torch.exp(current_action_log_probs - old_action_log_probs)
        advantage = batch_reward - self.get_values(batch_obs).squeeze()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        surr1 = ratio * (advantage.detach())
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * (advantage.detach())
        current_probs = torch.exp(current_log_probs)
        entropy = -torch.sum(current_probs * current_log_probs, dim=-1)

        policy_loss = -torch.min(surr1, surr2) - entropy * entropy_ratio
        value_loss = torch.pow(advantage, 2)
        return policy_loss.mean(), value_loss.mean(), entropy.mean().detach()


class VecEnvAgent(nn.Module):
    def __init__(self, p_net: PolicyNet):
        """
        An agent for acting in vectorized env.
        Args:
            p_net: The policy net.
        """
        super().__init__()
        self.net = p_net

    @torch.jit.export
    def get_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get probabilities for given obs
        Args:
            obs (torch.Tensor): The obs tensor, should be 2-dimensional

        Returns:
            torch.Tensor: The probabilities.
        """
        state_tensor = obs[:, :480]
        legal_actions = obs[:, 480:518]
        # print("legal_actions", legal_actions)
        probs = torch.exp(self.net(state_tensor))
        # print("probs", probs)
        return probs * legal_actions

    @torch.jit.export
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get actions for given obs
        Args:
            obs (torch.Tensor): The obs tensor, should be 2-dimensional

        Returns:
            torch.Tensor: The action

        """
        # vec env obs is always 2d
        probs = self.get_probs(obs)
        # print(probs)
        action = torch.argmax(probs, 1)
        return action


def random_vec_agent(device: str = "cuda") -> VecEnvAgent:
    """
    Get a random vec env agent.
    Args:
        device: The device of the agent.

    Returns:
        The agent.
    """
    net = PolicyNet()
    agent = VecEnvAgent(net)
    agent = agent.to(device)
    return agent
