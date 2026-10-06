#!/usr/bin/env python3

"""Rollout storage."""

import torch

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler


class RolloutStorage:

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        states_shape,
        actions_shape,
        device="cpu",
        sampler="sequential",
        recurrent_hidden_state_size=0,
    ):

        self.device = device
        self.sampler = sampler

        # Core
        self.observations = None
        self.states = torch.zeros(num_transitions_per_env, num_envs, states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.upward_action_supervision_masks = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.recurrent_hidden_states = None
        if recurrent_hidden_state_size > 0:
            self.recurrent_hidden_states = torch.zeros(
                num_transitions_per_env,
                num_envs,
                recurrent_hidden_state_size,
                device=self.device,
            )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(
        self, states, observations, actions, rewards, dones, values,
        actions_log_prob, mu, sigma, obs_device,
        upward_action_supervision_mask=None,
        recurrent_hidden_states=None,
    ):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        if observations is not None:
            if self.observations is None:
                self.observations = torch.zeros(
                    self.num_transitions_per_env, self.num_envs, *observations.shape[1:], device=obs_device
                )
            self.observations[self.step].copy_(observations.to(obs_device))
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)
        if upward_action_supervision_mask is None:
            self.upward_action_supervision_masks[self.step].zero_()
        else:
            self.upward_action_supervision_masks[self.step].copy_(
                upward_action_supervision_mask.detach().view(-1, 1)
            )
        if self.recurrent_hidden_states is not None:
            if recurrent_hidden_states is None:
                raise ValueError(
                    "Recurrent rollout storage requires hidden states"
                )
            self.recurrent_hidden_states[self.step].copy_(
                recurrent_hidden_states.detach()
            )

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch

    def recurrent_mini_batch_generator(self, num_mini_batches):
        if self.recurrent_hidden_states is None:
            raise RuntimeError(
                "Recurrent minibatches require recurrent rollout storage"
            )
        if self.num_envs % num_mini_batches != 0:
            raise ValueError(
                "num_envs must be divisible by num_mini_batches for "
                "recurrent PPO"
            )

        if self.sampler == "sequential":
            env_indices = torch.arange(self.num_envs)
        elif self.sampler == "random":
            env_indices = torch.randperm(self.num_envs)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")

        environments_per_batch = self.num_envs // num_mini_batches
        for start in range(0, self.num_envs, environments_per_batch):
            yield env_indices[
                start:start + environments_per_batch
            ].tolist()
