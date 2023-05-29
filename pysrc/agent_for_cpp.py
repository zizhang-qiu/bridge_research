"""
@file:agent_for_cpp
@author:qzz
@date:2023/2/23
@encoding:utf-8
"""
from typing import Optional, Tuple, Dict, List, Union

import torch
from torch import nn
import rl_cpp
from nets import PolicyNet, ValueNet, PerfectValueNet, PolicyNet2


class SingleEnvAgent(nn.Module):
    """An agent implementation which can be used in c++ by torch.jit.script"""

    def __init__(self, p_net: PolicyNet, v_net: Optional[ValueNet] = None):
        super().__init__()
        self.p_net = p_net
        self.v_net = v_net if v_net is not None else ValueNet()

    @torch.jit.export
    def get_probs(self, s: torch.Tensor) -> torch.Tensor:
        """
        Get probabilities for given obs
        Args:
            s (torch.Tensor): The obs tensor, should be 2-dimensional

        Returns:
            torch.Tensor: The probabilities.
        """
        probs = torch.exp(self.p_net(s))
        return probs

    @torch.jit.export
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get an action for given obs
        Args:
            obs (torch.Tensor): The obs tensor, should be 1-dimensional

        Returns:
            torch.Tensor: The action

        """
        # single env obs is always 1d
        greedy = obs["greedy"].item()
        legal_actions = obs["legal_actions"]
        s = obs["s"]
        log_probs = self.get_log_probs(s.unsqueeze(0)).squeeze()
        probs = torch.exp(log_probs) * legal_actions
        value = self.v_net(s).squeeze()
        # print(probs)
        if greedy:
            action = torch.argmax(probs)
        else:
            if torch.equal(probs, torch.zeros_like(probs)):
                print("Warning: all the probs are zero")
                action = torch.multinomial(legal_actions, 1).squeeze()
            else:
                action = torch.multinomial(probs, 1).squeeze()
        return {"a": action.detach().cpu(), "log_probs": log_probs.detach().cpu(), "values": value}

    @torch.jit.export
    def get_log_probs(self, s: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities for given obs
        Args:
            s (torch.Tensor): The obs tensor, should be 2-dimensional

        Returns:
            torch.Tensor: The log probabilities
        """
        log_probs = self.p_net(s)
        return log_probs

    @torch.jit.export
    def get_top_k_actions_with_min_prob(self, obs: Dict[str, torch.Tensor], top_k: int, min_prob: float) \
            -> Dict[str, torch.Tensor]:
        """
        Get top k actions with at least min prob
        Args:
            obs: The obs tensordict
            top_k: How many actions to get
            min_prob: The minimum prob

        Returns:
            The top k actions
        """
        legal_actions = obs["legal_actions"]
        s = obs["s"]
        log_probs = self.get_log_probs(s.unsqueeze(0)).squeeze()
        probs = torch.exp(log_probs) * legal_actions
        probs, indices = torch.topk(probs, top_k)
        available_indices = probs > min_prob
        top_k_with_min_prob = indices[available_indices].int()
        top_k_probs = probs[available_indices]
        return {"top_k_actions": top_k_with_min_prob, "top_k_probs": top_k_probs}

    @torch.jit.export
    def get_prob_for_action(self, obs: Dict[str, torch.Tensor], action: int) -> float:
        legal_actions = obs["legal_actions"]
        s = obs["s"]
        log_probs = self.get_log_probs(s.unsqueeze(0)).squeeze()
        probs = torch.exp(log_probs) * legal_actions
        action_prob = probs[action].item()
        return action_prob

    def simple_act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        legal_actions = obs["legal_actions"]
        s = obs["s"]
        log_probs = self.get_log_probs(s.unsqueeze(0)).squeeze()
        raw_probs = torch.exp(log_probs)
        probs = raw_probs * legal_actions
        action = torch.argmax(probs)
        return {"a": action.detach().cpu(), "probs": probs.detach().cpu(), "raw_probs": raw_probs.detach().cpu()}

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
        advantage = batch_reward
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
    def __init__(self, p_net: Union[PolicyNet, PolicyNet2]):
        """
        An agent for acting in vectorized env.
        Args:
            p_net: The policy net.
        """
        super().__init__()
        self.p_net = p_net
        self.v_net = PerfectValueNet()

    def compute_priority(self, batch: rl_cpp.Transition):
        values = self.get_values(batch.obs["s"]).squeeze()
        rewards = batch.reward
        td_err = rewards - values
        return torch.abs(td_err).detach().cpu()

    @torch.jit.export
    def get_probs(self, s: torch.Tensor) -> torch.Tensor:
        """
        Get probabilities for given obs
        Args:
            s (torch.Tensor): The obs tensor, should be 2-dimensional

        Returns:
            torch.Tensor: The probabilities.
        """
        # print("legal_actions", legal_actions)
        probs = torch.exp(self.p_net(s))
        # print("probs", probs)
        return probs

    def get_values(self, s: torch.Tensor):
        values = self.v_net(s)
        return values

    @torch.jit.export
    def get_log_probs(self, s: torch.Tensor) -> torch.Tensor:
        probs = self.p_net(s)
        return probs

    @torch.jit.export
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get actions for given obs
        Args:
            obs (torch.Tensor): The obs tensor, should be 2-dimensional

        Returns:
            torch.Tensor: The action

        """
        # vec env obs is always 2d
        s = obs["s"]
        perfect_s = obs["perfect_s"]
        legal_actions = obs["legal_actions"]
        log_probs = self.get_log_probs(s)
        probs = torch.exp(log_probs)
        topk_values, topk_indices = probs.topk(4, dim=1)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, topk_indices, 1)
        legal_probs = probs * legal_actions
        all_zeros = torch.all(legal_probs == 0, dim=1)
        zero_indices = torch.nonzero(all_zeros).squeeze()
        if zero_indices.numel() > 0:
            print("warning.")
            legal_probs[zero_indices] = legal_actions[zero_indices]
        greedy = obs["greedy"]

        greedy_action = torch.argmax(legal_probs, 1)
        # print("greedy actions",  greedy_action)
        random_actions = torch.multinomial(legal_probs, 1).squeeze()

        # print("random actions", random_actions)
        action = greedy * greedy_action + (1 - greedy) * random_actions
        values = self.v_net(perfect_s).squeeze()
        return {"a": action.detach().cpu(), "log_probs": log_probs.detach().cpu(), "values": values.detach().cpu()}

    def compute_loss_and_priority(self, batch: rl_cpp.Transition, clip_eps: float, entropy_ratio: float) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_log_probs = self.get_log_probs(batch.obs["s"])
        current_action_log_probs = current_log_probs.gather(1, batch.reply["a"].long().unsqueeze(1)).squeeze(1)
        old_action_log_probs = batch.reply["log_probs"].gather(1, batch.reply["a"].long().unsqueeze(1)).squeeze(1)

        ratio = torch.exp(current_action_log_probs - old_action_log_probs)
        # print("ratio:", ratio)
        # advantage = (batch_reward - batch_reward.mean()) / (batch_reward.std() + 1e-5)
        # advantage = batch_reward
        values = self.get_values(batch.obs["perfect_s"]).squeeze()
        advantage = batch.reward - values
        # print("advantage:", advantage)

        surr1 = ratio * (advantage.detach())
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * (advantage.detach())
        current_probs = torch.exp(current_log_probs)
        entropy = -torch.sum(current_probs * current_log_probs, dim=-1)
        # print("entropy:", entropy)

        policy_loss = -torch.min(surr1, surr2) - entropy * entropy_ratio
        value_loss = torch.pow(advantage, 2)
        rewards = batch.reward
        td_err = rewards - values
        return policy_loss, value_loss, torch.abs(td_err).detach().cpu()

    def pg_loss(self, batch_state: torch.Tensor, batch_action: torch.Tensor,
                batch_reward: torch.Tensor, batch_log_probs: torch.Tensor,
                clip_eps: float, entropy_ratio: float):
        current_log_probs = self.get_log_probs(batch_state)
        current_action_log_probs = current_log_probs.gather(1, batch_action.long().unsqueeze(1)).squeeze(1)
        old_action_log_probs = batch_log_probs.gather(1, batch_action.long().unsqueeze(1)).squeeze(1)

        ratio = torch.exp(current_action_log_probs - old_action_log_probs)
        # advantage = (batch_reward - batch_reward.mean()) / (batch_reward.std() + 1e-5)
        # advantage = batch_reward

        surr1 = ratio * (batch_reward.detach())
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * (batch_reward.detach())
        current_probs = torch.exp(current_log_probs)
        entropy = -torch.sum(current_probs * current_log_probs, dim=-1)

        policy_loss = -torch.min(surr1, surr2) - entropy * entropy_ratio
        return policy_loss.mean()


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
