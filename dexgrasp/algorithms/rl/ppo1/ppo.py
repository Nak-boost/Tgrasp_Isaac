import pickle
import json
from datetime import datetime
import os
import time
import math
import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import write_video
from algorithms.rl.ppo1 import RolloutStorage
from algorithms.rl.ppo1.policy import ActorCriticDexRep, ActorCriticMultimodal


import copy
from utils.tensorboard_extract import tensorboard2csv

_MODEL_FUNCS = {
    "ActorCriticDexRep": ActorCriticDexRep,
    "ActorCriticMultimodal": ActorCriticMultimodal,
}

_TACTILE_EPISODE_METRICS = {
    "contact_found": ("Contact discovery rate", 100.0),
    "first_contact_step": ("Mean first-contact step", 1.0),
    "multi_contact": ("Multi-contact episode rate", 100.0),
    "lifted": ("Lift episode rate", 100.0),
    "contact_step_ratio": ("Contact-step ratio", 100.0),
    "contact_losses": ("Contact losses per episode", 1.0),
}

_UPWARD_SUPERVISION_MASK = "upward_action_supervision_mask"
_CURRENT_CONTACT_MASK = "current_contact"
_VALID_GRASP_MASK = "valid_grasp"
_TACTILE_STEP_METRICS = {
    _UPWARD_SUPERVISION_MASK,
    _CURRENT_CONTACT_MASK,
    _VALID_GRASP_MASK,
}
_NOISE_PHASES = (
    "no_contact",
    "contact_invalid",
    "valid_grasp",
)
_SHADOW_HAND_ACTION_NAMES = (
    "palm_force_x",
    "palm_force_y",
    "palm_force_z",
    "palm_torque_x",
    "palm_torque_y",
    "palm_torque_z",
    "ffj3",
    "ffj2",
    "ffj1",
    "mfj3",
    "mfj2",
    "mfj1",
    "rfj3",
    "rfj2",
    "rfj1",
    "lfj4",
    "lfj3",
    "lfj2",
    "lfj1",
    "thj4",
    "thj3",
    "thj2",
    "thj1",
    "thj0",
)


class PPO:
    def __init__(self,
                 vec_env,
                 cfg_train,
                 cfg_env,
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 device='cpu',
                 ):

        # if not isinstance(vec_env.observation_space, Space):
        #     raise TypeError("vec_env.observation_space must be a gym Space")
        # if not isinstance(vec_env.action_space, Space):
        #     raise TypeError("vec_env.action_space must be a gym Space")
        self.state_space = vec_env.observation_space
        self.action_space = vec_env.action_space

        self.cfg_train = copy.deepcopy(cfg_train)
        self.cfg_env = copy.deepcopy(cfg_env)
        self.tactile_experiment = self.cfg_env.get(
            "tactile", {}
        ).get("experiment")
        learn_cfg = self.cfg_train["learn"]
        self.device = torch.device(device)

        self.desired_kl = learn_cfg.get("desired_kl", None)
        self.schedule = learn_cfg.get("schedule", "fixed")
        self.step_size = learn_cfg["optim_stepsize"]
        self.init_noise_std = learn_cfg.get("init_noise_std", 0.3)
        self.model_cfg = self.cfg_train["policy"]
        self.encoder_cfg = self.cfg_train['encoder']
        self.num_transitions_per_env=learn_cfg["nsteps"]
        # self.num_transitions_per_env = self.cfg_env['ep_length']  # do rollout each episode
        self.learning_rate=learn_cfg["optim_stepsize"]
        self.max_len=learn_cfg.get("max_len", 200)
        self.step_size_init = self.learning_rate

        # PPO components
        self.vec_env = vec_env
        actor_critic_name = self.model_cfg["actor_critic"]
        if actor_critic_name not in _MODEL_FUNCS:
            raise ValueError(f"Unknown actor critic model: {actor_critic_name}")
        actor_critic_class = _MODEL_FUNCS[actor_critic_name]
        self.actor_critic = actor_critic_class(
            self.state_space.shape,
            self.action_space.shape,
            self.init_noise_std,
            self.model_cfg,
            self.encoder_cfg,
            self.cfg_env,
        )
        self.actor_critic.to(self.device)
        self.is_recurrent = bool(
            getattr(self.actor_critic, "is_recurrent", False)
        )
        self.recurrent_hidden_size = int(
            getattr(self.actor_critic, "recurrent_hidden_size", 0)
            if self.is_recurrent else 0
        )
        print(f"Encoder Name: {actor_critic_name}")
        if hasattr(self.actor_critic, "initial_action_std"):
            print(
                "Action noise std: "
                f"initial={self.actor_critic.initial_action_std:.3f}, "
                f"range=[{self.actor_critic.minimum_action_std:.3f}, "
                f"{self.actor_critic.maximum_action_std:.3f}]"
            )

        self.obs_device = self.device if self.encoder_cfg['name'] not in ['resnet18'] else 'cpu'
        self.storage = RolloutStorage(
            self.vec_env.num_envs,
            self.num_transitions_per_env,
            None,
            self.cfg_env['obs_dim']['prop'],
            self.action_space.shape,
            self.device,
            sampler,
            recurrent_hidden_state_size=self.recurrent_hidden_size,
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

        # PPO parameters
        self.clip_param = learn_cfg["cliprange"]
        self.num_learning_epochs = learn_cfg["noptepochs"]
        self.num_mini_batches = learn_cfg["nminibatches"]
        # self.num_transitions_per_env = self.num_transitions_per_env
        self.value_loss_coef = learn_cfg.get("value_loss_coef", 2.0)
        self.entropy_coef = learn_cfg["ent_coef"]
        self.gamma = learn_cfg["gamma"]
        self.lam = learn_cfg["lam"]
        self.max_grad_norm = learn_cfg.get("max_grad_norm", 2.0)
        self.use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False)
        supervision_cfg = learn_cfg.get("upward_action_supervision", {})
        self.upward_supervision_enabled = bool(
            supervision_cfg.get("enabled", False)
        )
        self.upward_supervision_target = float(
            supervision_cfg.get("target_action", 0.3)
        )
        self.upward_supervision_index = int(
            supervision_cfg.get("action_index", 2)
        )
        self.upward_supervision_coef = float(
            supervision_cfg.get("loss_coef", 0.1)
        )
        self.upward_supervision_trigger_rate = float(
            supervision_cfg.get("trigger_lift_rate", 0.25)
        )
        self.upward_supervision_window = int(
            supervision_cfg.get("success_window", 2000)
        )
        self.upward_supervision_decay_iterations = int(
            supervision_cfg.get("decay_iterations", 2000)
        )
        if self.upward_supervision_enabled:
            if not self.cfg_env.get("tactile", {}).get("enabled", False):
                raise ValueError("Upward supervision requires tactile observations")
            if self.cfg_env.get("useRelativeControl", False):
                raise ValueError("Upward supervision requires hand-base force control")
            if not 0 <= self.upward_supervision_index < self.action_space.shape[0]:
                raise ValueError("upward_action_supervision.action_index is out of range")
            if not 0.0 < self.upward_supervision_target <= self.cfg_train.get("clip_actions", 1.0):
                raise ValueError("upward_action_supervision.target_action must be positive and within clip_actions")
            if not math.isfinite(self.upward_supervision_coef) or self.upward_supervision_coef < 0.0:
                raise ValueError("upward_action_supervision.loss_coef must be finite and nonnegative")
            if not 0.0 < self.upward_supervision_trigger_rate <= 1.0:
                raise ValueError("upward_action_supervision.trigger_lift_rate must be in (0, 1]")
            if self.upward_supervision_window < 1 or self.upward_supervision_decay_iterations < 1:
                raise ValueError("Supervision success_window and decay_iterations must be positive")
        self.upward_supervision_decay_start = None
        self.upward_supervision_lift_buffer = deque(
            maxlen=max(1, self.upward_supervision_window)
        )
        self.upward_supervision_metrics = {}
        self.model_dir = log_dir+'/checkpoint'
        os.makedirs(self.model_dir, exist_ok=True)

        # Log
        self.log_dir = log_dir+'/logger'
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()
        if self.upward_supervision_enabled:
            self.upward_supervision_decay_start = None
            self.upward_supervision_lift_buffer.clear()
            schedule_path = path + ".supervision.json"
            if os.path.isfile(schedule_path):
                with open(schedule_path) as handle:
                    schedule = json.load(handle)
                self.upward_supervision_decay_start = schedule["decay_start"]
                self.upward_supervision_lift_buffer.extend(schedule["lifted_episodes"])
            else:
                print("Weights-only checkpoint: upward supervision schedule starts fresh")

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)
        if self.upward_supervision_enabled:
            with open(path + ".supervision.json", "w") as handle:
                json.dump({
                    "decay_start": self.upward_supervision_decay_start,
                    "lifted_episodes": list(self.upward_supervision_lift_buffer),
                }, handle)

    def get_upward_supervision_coef(self, iteration):
        if not self.upward_supervision_enabled:
            return 0.0
        if self.upward_supervision_decay_start is None:
            return self.upward_supervision_coef
        remaining = max(
            1.0 - (iteration - self.upward_supervision_decay_start)
            / self.upward_supervision_decay_iterations,
            0.0,
        )
        return self.upward_supervision_coef * min(remaining, 1.0)

    def update_upward_supervision_schedule(self, iteration, lifted_episodes):
        if not self.upward_supervision_enabled:
            return
        self.upward_supervision_lift_buffer.extend(lifted_episodes)
        if (
            self.upward_supervision_decay_start is None
            and len(self.upward_supervision_lift_buffer) == self.upward_supervision_window
            and statistics.mean(self.upward_supervision_lift_buffer)
            >= self.upward_supervision_trigger_rate
        ):
            self.upward_supervision_decay_start = iteration
            print(f"Upward supervision decay starts at iteration {iteration}")

    def compute_upward_supervision_loss(self, action_means, mask):
        weights = mask.detach().reshape(-1).to(action_means.dtype)
        deficit = torch.relu(
            self.upward_supervision_target
            - action_means[:, self.upward_supervision_index]
        )
        return (deficit.square() * weights).sum() / weights.sum().clamp(min=1.0)

    def initial_recurrent_state(self):
        if not self.is_recurrent:
            return None
        return self.actor_critic.initial_recurrent_state(
            self.vec_env.num_envs,
            self.device,
        )

    def act(self, observations, recurrent_hidden_states):
        if self.is_recurrent:
            return self.actor_critic.act(
                observations,
                recurrent_hidden_states,
            )
        return self.actor_critic.act(observations) + (None,)

    def act_inference(self, observations, recurrent_hidden_states):
        if self.is_recurrent:
            return self.actor_critic.act_inference(
                observations,
                recurrent_hidden_states,
            )
        return self.actor_critic.act_inference(observations), None

    def mask_recurrent_state(self, recurrent_hidden_states, dones):
        if recurrent_hidden_states is None:
            return None
        return recurrent_hidden_states * (
            dones.reshape(-1, 1).to(self.device) == 0
        ).to(recurrent_hidden_states.dtype)

    def get_last_values(self, observations, recurrent_hidden_states):
        if self.is_recurrent:
            return self.actor_critic.get_value(
                observations,
                recurrent_hidden_states,
            )
        return self.actor_critic.act(observations)[2]

    @staticmethod
    def collect_terminal_metrics(metric_values, infos, env_ids):
        for metric_name in _TACTILE_EPISODE_METRICS:
            if metric_name not in infos:
                continue
            values = (
                infos[metric_name][env_ids]
                .reshape(-1)
                .detach()
                .cpu()
                .tolist()
            )
            if metric_name == "first_contact_step":
                values = [value for value in values if value >= 0]
            metric_values[metric_name].extend(values)

    @staticmethod
    def print_terminal_metrics(metric_values, prefix=""):
        available_metrics = [
            metric_name
            for metric_name, values in metric_values.items()
            if values
        ]
        if not available_metrics:
            return

        if prefix:
            print(prefix)
        for metric_name in available_metrics:
            label, scale = _TACTILE_EPISODE_METRICS[metric_name]
            value = statistics.mean(metric_values[metric_name]) * scale
            suffix = "%" if scale == 100.0 else ""
            print(f"{label}: {value:.2f}{suffix}")

    def run(self, num_learning_iterations, log_interval=1):
        # current_obs_state = self.vec_env.reset()
        # current_states = self.vec_env.get_state()
        current_obs_state = self.vec_env.reset()
        recurrent_hidden_states = self.initial_recurrent_state()
        if self.is_testing:
            maxlen = 100
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []
            successes = []
            successes_times = []
            episode_metrics = {
                metric_name: []
                for metric_name in _TACTILE_EPISODE_METRICS
            }
            recoder = dict(images=[], tactiles=[])
            current_obs_state = self.vec_env.reset()
            recurrent_hidden_states = self.initial_recurrent_state()
            while len(reward_sum) <= maxlen:
                # recoder["images"].append(current_obs_state[:, 50:-20].cpu().numpy().reshape(-1, 224, 224, 3))
                recoder["tactiles"].append(current_obs_state[:, -20:].cpu().numpy())
                with torch.no_grad():
                    # Compute the action
                    actions, next_recurrent_hidden_states = (
                        self.act_inference(
                            current_obs_state,
                            recurrent_hidden_states,
                        )
                    )
                    # Step the vec_environment
                    ALL_RESULTS = self.vec_env.step(actions)
                    if len(ALL_RESULTS) == 4:
                        next_obs_state, rews, dones, infos = ALL_RESULTS
                        self.enable_success_time = False
                    elif len(ALL_RESULTS) == 5:
                        next_obs_state, rews, dones, success_time, infos = ALL_RESULTS
                        self.enable_success_time = True
                    else:
                        raise KeyError("")
                    # self.vec_env.render()
                    current_obs_state.copy_(next_obs_state)
                    recurrent_hidden_states = self.mask_recurrent_state(
                        next_recurrent_hidden_states,
                        dones,
                    )
                    cur_reward_sum[:] += rews
                    cur_episode_length[:] += 1

                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    success_key = (
                        "successes"
                        if "successes" in infos
                        else "goal_achieved"
                    )
                    successes.extend(
                        infos[success_key][new_ids]
                        .reshape(-1).cpu().numpy().tolist()
                    )
                    self.collect_terminal_metrics(
                        episode_metrics, infos, new_ids
                    )
                    if self.enable_success_time:
                        successes_times.extend(success_time[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                    if len(new_ids) > 0:
                        print("-" * 80)
                        print("Num episodes: {}".format(len(reward_sum)))
                        print("Mean return: {:.2f}".format(statistics.mean(reward_sum)))
                        print("Mean ep len: {:.2f}".format(statistics.mean(episode_length)))
                        print("Mean success: {:.2f}".format(statistics.mean(successes) * 100))
                        if self.enable_success_time:
                            print("Mean success time: {:.2f}".format(statistics.mean(successes_times)))
                        self.print_terminal_metrics(
                            episode_metrics,
                            prefix=(
                                f"Tactile experiment: "
                                f"{self.tactile_experiment}"
                            ),
                        )
                        # pickle.dump(recoder, open(f'./I&T{len(reward_sum)}.pkl', 'wb'))
                        recoder = dict(images=[], tactiles=[])
                # print(f"sum of imges is {len(recoder['images'])}")

        else:
            rewbuffer = deque(maxlen=self.max_len)
            lenbuffer = deque(maxlen=self.max_len)
            successbuffer = deque(maxlen=self.max_len)
            successes_time_buffer = deque(maxlen=self.max_len)
            episode_metric_buffers = {
                metric_name: deque(maxlen=self.max_len)
                for metric_name in _TACTILE_EPISODE_METRICS
            }
            # print(self.max_len)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            env_mean_success = 0.0
            supervision_mask = torch.zeros(
                self.vec_env.num_envs, 1, device=self.device
            )
            current_contact_mask = torch.zeros(
                self.vec_env.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
            valid_grasp_mask = torch.zeros_like(current_contact_mask)
            action_count = self.action_space.shape[0]
            action_names = (
                _SHADOW_HAND_ACTION_NAMES
                if action_count == len(_SHADOW_HAND_ACTION_NAMES)
                else tuple(
                    f"action_{index:02d}"
                    for index in range(action_count)
                )
            )

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []
                reward_sum = []
                episode_length = []
                successes = []
                successes_times = []
                episode_metrics = {
                    metric_name: []
                    for metric_name in _TACTILE_EPISODE_METRICS
                }
                lifted_episodes = []
                noise_phase_sums = {
                    phase: torch.zeros(
                        action_count,
                        device=self.device,
                    )
                    for phase in _NOISE_PHASES
                }
                noise_phase_counts = {
                    phase: torch.zeros((), device=self.device)
                    for phase in _NOISE_PHASES
                }

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs_state = self.vec_env.reset()
                        recurrent_hidden_states = (
                            self.initial_recurrent_state()
                        )
                        supervision_mask.zero_()
                        current_contact_mask.zero_()
                        valid_grasp_mask.zero_()
                    # Compute the action
                    (
                        actions,
                        actions_log_prob,
                        values,
                        mu,
                        sigma,
                        current_state,
                        current_obs_feats,
                        next_recurrent_hidden_states,
                    ) = self.act(
                        current_obs_state,
                        recurrent_hidden_states,
                    )
                    action_std = sigma.exp()
                    phase_masks = {
                        "no_contact": torch.logical_not(
                            current_contact_mask
                        ),
                        "contact_invalid": torch.logical_and(
                            current_contact_mask,
                            torch.logical_not(valid_grasp_mask),
                        ),
                        "valid_grasp": valid_grasp_mask,
                    }
                    for phase, phase_mask in phase_masks.items():
                        phase_weights = phase_mask.to(
                            action_std.dtype
                        ).unsqueeze(1)
                        noise_phase_sums[phase].add_(
                            (action_std * phase_weights).sum(dim=0)
                        )
                        noise_phase_counts[phase].add_(
                            phase_weights.sum()
                        )
                    # Step the vec_environment
                    ALL_RESULTS = self.vec_env.step(actions)
                    if len(ALL_RESULTS) == 4:
                        next_obs_state, rews, dones, infos = ALL_RESULTS
                        self.enable_success_time = False
                    elif len(ALL_RESULTS) == 5:
                        next_obs_state, rews, dones, success_time, infos = ALL_RESULTS
                        self.enable_success_time = True
                    else:
                        raise KeyError("")
                    # print(rews)
                    # next_states = self.vec_env.get_state()
                    # Record the transition
                    self.storage.add_transitions(
                        current_state, current_obs_feats, actions, rews, dones,
                        values, actions_log_prob, mu, sigma, self.obs_device,
                        supervision_mask,
                        recurrent_hidden_states,
                    )
                    if self.upward_supervision_enabled:
                        if _UPWARD_SUPERVISION_MASK not in infos or "lifted" not in infos:
                            raise KeyError("Environment is missing upward supervision metrics")
                        supervision_mask.copy_(
                            infos[_UPWARD_SUPERVISION_MASK].reshape(-1, 1).to(self.device)
                        )
                        supervision_mask.mul_(
                            (dones.reshape(-1, 1).to(self.device) == 0).float()
                        )
                        terminal = dones.reshape(-1).to(infos["lifted"].device) > 0
                        lifted_episodes.extend(
                            infos["lifted"].reshape(-1)[terminal].detach().cpu().tolist()
                        )
                    if self.tactile_experiment:
                        missing_phase_metrics = (
                            {_CURRENT_CONTACT_MASK, _VALID_GRASP_MASK}
                            - set(infos)
                        )
                        if missing_phase_metrics:
                            raise KeyError(
                                "Environment is missing noise phase metrics: "
                                f"{sorted(missing_phase_metrics)}"
                            )
                        active = (
                            dones.reshape(-1).to(self.device) == 0
                        )
                        current_contact_mask.copy_(
                            (
                                infos[_CURRENT_CONTACT_MASK]
                                .reshape(-1)
                                .to(self.device)
                                > 0.5
                            )
                            & active
                        )
                        valid_grasp_mask.copy_(
                            (
                                infos[_VALID_GRASP_MASK]
                                .reshape(-1)
                                .to(self.device)
                                > 0.5
                            )
                            & active
                        )
                    current_obs_state.copy_(next_obs_state)
                    recurrent_hidden_states = self.mask_recurrent_state(
                        next_recurrent_hidden_states,
                        dones,
                    )
                    # current_states.copy_(next_states)
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # if len(new_ids)>0:
                        #     pass
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        successes.extend(infos['successes'][new_ids][:, 0].cpu().numpy().tolist())
                        self.collect_terminal_metrics(
                            episode_metrics, infos, new_ids
                        )
                        if self.enable_success_time:
                            successes_times.extend(success_time[new_ids][:, 0].cpu().numpy().tolist())
                        # successes.extend(infos['env_mean_successes'][new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        env_mean_success = 0
                        if infos.get("env_mean_successes"):
                            env_mean_success = infos['consecutive_successes'].mean()*100
                # sr statistics
                # all_sr = np.array(successes).reshape(self.num_transitions_per_env,self.vec_env.num_envs)
                # all_sr = np.sum(all_sr, axis=0) > 1

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)
                    successbuffer.extend(successes)
                    if self.enable_success_time:
                        successes_time_buffer.extend(successes_times)
                    for metric_name, values in episode_metrics.items():
                        episode_metric_buffers[metric_name].extend(
                            values
                        )

                last_values = self.get_last_values(
                    current_obs_state,
                    recurrent_hidden_states,
                )
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()
                rollout_noise_std = self.storage.sigma.exp()
                mean_noise_std = rollout_noise_std.mean().item()
                minimum_noise_std = rollout_noise_std.min().item()
                maximum_noise_std = rollout_noise_std.max().item()
                total_noise_samples = float(
                    self.num_transitions_per_env
                    * self.vec_env.num_envs
                )
                noise_phase_metrics = {}
                noise_phase_summary = {}
                for phase in _NOISE_PHASES:
                    count = noise_phase_counts[phase]
                    phase_action_std = (
                        noise_phase_sums[phase]
                        / count.clamp(min=1.0)
                    )
                    fraction = count.item() / total_noise_samples
                    mean_std = phase_action_std.mean().item()
                    prefix = f"Policy/noise_phase/{phase}"
                    noise_phase_metrics[f"{prefix}/sample_fraction"] = (
                        fraction
                    )
                    noise_phase_metrics[f"{prefix}/mean"] = mean_std
                    noise_phase_metrics[
                        f"{prefix}/palm_translation"
                    ] = phase_action_std[:3].mean().item()
                    noise_phase_metrics[
                        f"{prefix}/palm_rotation"
                    ] = phase_action_std[3:6].mean().item()
                    if action_count > 6:
                        noise_phase_metrics[
                            f"{prefix}/finger_joints"
                        ] = phase_action_std[6:].mean().item()
                    for action_name, action_value in zip(
                        action_names,
                        phase_action_std.tolist(),
                    ):
                        noise_phase_metrics[
                            f"{prefix}/{action_name}"
                        ] = action_value
                    noise_phase_summary[phase] = (
                        fraction,
                        mean_std,
                    )

                # Learning step
                start = stop
                self.update_upward_supervision_schedule(it, lifted_episodes)
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update(it, num_learning_iterations)

                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % (log_interval/2) == 0:
                    # self.save(os.path.join(self.model_dir, 'model_{}.pt'.format(it)))
                    # # add 1129 for extract results in tensorboard online
                    self.results2cvs()
                if it % (2 * log_interval) == 0:
                    self.save(os.path.join(self.model_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.model_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                if (
                    key in _TACTILE_EPISODE_METRICS
                    or key in _TACTILE_STEP_METRICS
                ):
                    continue
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = locs["mean_noise_std"]

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        for name, value in self.upward_supervision_metrics.items():
            self.writer.add_scalar(name, value, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std, locs['it'])
        self.writer.add_scalar(
            'Policy/min_noise_std',
            locs['minimum_noise_std'],
            locs['it'],
        )
        self.writer.add_scalar(
            'Policy/max_noise_std',
            locs['maximum_noise_std'],
            locs['it'],
        )
        for name, value in locs["noise_phase_metrics"].items():
            self.writer.add_scalar(name, value, locs["it"])
        self.writer.add_scalar('Train/env_mean_success', locs['env_mean_success'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_success', statistics.mean(locs['successbuffer'])*100, locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_success/time', statistics.mean(locs['successbuffer'])*100, self.tot_time)
            if self.enable_success_time:
                self.writer.add_scalar('Train/mean_success_time', statistics.mean(locs['successes_time_buffer']), locs['it'])

        tactile_string = ""
        for metric_name, values in locs[
            "episode_metric_buffers"
        ].items():
            if not values:
                continue
            label, scale = _TACTILE_EPISODE_METRICS[metric_name]
            value = statistics.mean(values) * scale
            self.writer.add_scalar(
                "Tactile/" + metric_name,
                value,
                locs["it"],
            )
            suffix = "%" if scale == 100.0 else ""
            tactile_string += (
                f"{label + ':':>{pad}} {value:.2f}{suffix}\n"
            )

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'task_name: ':>{pad}} {self.encoder_cfg['name']}\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""
                          f"""{'Action noise std range:':>{pad}} {locs['minimum_noise_std']:.2f} - {locs['maximum_noise_std']:.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean success rate:':>{pad}} {statistics.mean(locs['successbuffer'])*100:.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
            if self.enable_success_time:
                log_string += (
                    f"""{'Mean success time:':>{pad}} {statistics.mean(locs['successes_time_buffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'task_name: ':>{pad}} {self.encoder_cfg['name']}\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""
                          f"""{'Action noise std range:':>{pad}} {locs['minimum_noise_std']:.2f} - {locs['maximum_noise_std']:.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        log_string += (f"""{'env_mean_success':>{pad}} {locs['env_mean_success']:.4f}\n""")
        if self.tactile_experiment:
            log_string += (
                f"{'Tactile experiment:':>{pad}} "
                f"{self.tactile_experiment}\n"
            )
            for phase in _NOISE_PHASES:
                fraction, phase_mean = locs[
                    "noise_phase_summary"
                ][phase]
                log_string += (
                    f"{('Noise ' + phase + ':'):>{pad}} "
                    f"std={phase_mean:.3f}, "
                    f"samples={fraction * 100.0:.1f}%\n"
                )
        log_string += tactile_string
        if self.upward_supervision_enabled:
            log_string += (
                f"{'Upward supervision loss:':>{pad}} "
                f"{self.upward_supervision_metrics['Loss/upward_action_supervision']:.4f}\n"
                f"{'Upward supervision coefficient:':>{pad}} "
                f"{self.get_upward_supervision_coef(locs['it']):.4f}\n"
            )
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def results2cvs(self):

        out_path = os.path.dirname(self.log_dir) + "/log_train.csv"
        tensorboard2csv(self.log_dir, out_path)
        print(f"Generate results to {out_path}")

    def update(self, cur_iter, max_iter):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_supervision_loss = 0.0
        supervision_coef = self.get_upward_supervision_coef(cur_iter)

        for epoch in range(self.num_learning_epochs):
            if self.schedule == "cos":
                self.step_size = self.adjust_learning_rate_cos(
                    self.optimizer, epoch, self.num_learning_epochs, cur_iter, max_iter
                )
            if self.is_recurrent:
                batch = self.storage.recurrent_mini_batch_generator(
                    self.num_mini_batches
                )
            else:
                batch = self.storage.mini_batch_generator(
                    self.num_mini_batches
                )

            for indices in batch:
                if self.is_recurrent:
                    obs_sequence = (
                        self.storage.observations[:, indices]
                        .to(self.device)
                    )
                    states_sequence = self.storage.states[:, indices]
                    actions_sequence = self.storage.actions[:, indices]
                    dones_sequence = self.storage.dones[:, indices]
                    initial_hidden_states = (
                        self.storage.recurrent_hidden_states[0, indices]
                    )
                    (
                        actions_log_prob_batch,
                        entropy_batch,
                        value_batch,
                        mu_batch,
                        sigma_batch,
                    ) = self.actor_critic.evaluate_recurrent(
                        obs_sequence,
                        states_sequence,
                        actions_sequence,
                        initial_hidden_states,
                        dones_sequence,
                    )
                    actions_batch = actions_sequence.reshape(
                        -1, self.storage.actions.size(-1)
                    )
                    target_values_batch = self.storage.values[
                        :, indices
                    ].reshape(-1, 1)
                    returns_batch = self.storage.returns[
                        :, indices
                    ].reshape(-1, 1)
                    old_actions_log_prob_batch = (
                        self.storage.actions_log_prob[:, indices]
                        .reshape(-1, 1)
                    )
                    advantages_batch = self.storage.advantages[
                        :, indices
                    ].reshape(-1, 1)
                    old_mu_batch = self.storage.mu[
                        :, indices
                    ].reshape(-1, self.storage.actions.size(-1))
                    old_sigma_batch = self.storage.sigma[
                        :, indices
                    ].reshape(-1, self.storage.actions.size(-1))
                    supervision_mask_batch = (
                        self.storage.upward_action_supervision_masks[
                            :, indices
                        ].reshape(-1, 1)
                    )
                else:
                    obs_batch = (
                        self.storage.observations.view(
                            -1, *self.storage.observations.size()[2:]
                        )[indices].to(self.device)
                        if self.storage.observations is not None
                        else None
                    )
                    states_batch = self.storage.states.view(
                        -1, *self.storage.states.size()[2:]
                    )[indices]
                    actions_batch = self.storage.actions.view(
                        -1, self.storage.actions.size(-1)
                    )[indices]
                    target_values_batch = self.storage.values.view(
                        -1, 1
                    )[indices]
                    returns_batch = self.storage.returns.view(
                        -1, 1
                    )[indices]
                    old_actions_log_prob_batch = (
                        self.storage.actions_log_prob.view(-1, 1)[indices]
                    )
                    advantages_batch = self.storage.advantages.view(
                        -1, 1
                    )[indices]
                    old_mu_batch = self.storage.mu.view(
                        -1, self.storage.actions.size(-1)
                    )[indices]
                    old_sigma_batch = self.storage.sigma.view(
                        -1, self.storage.actions.size(-1)
                    )[indices]
                    supervision_mask_batch = (
                        self.storage.upward_action_supervision_masks
                        .view(-1, 1)[indices]
                    )
                    (
                        actions_log_prob_batch,
                        entropy_batch,
                        value_batch,
                        mu_batch,
                        sigma_batch,
                    ) = self.actor_critic.evaluate(
                        obs_batch,
                        states_batch,
                        actions_batch,
                    )

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':

                    kl = torch.sum(sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                    # print(f'returns_batch: {returns_batch}')
                    # print(f'returns_batch: {returns_batch}')

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                if self.upward_supervision_enabled:
                    supervision_loss = self.compute_upward_supervision_loss(
                        mu_batch,
                        supervision_mask_batch,
                    )
                    loss = loss + supervision_coef * supervision_loss
                    mean_supervision_loss += supervision_loss.item()
                # print(loss.item())
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        if self.upward_supervision_enabled:
            masks = self.storage.upward_action_supervision_masks
            selected_means = self.storage.mu[:, :, self.upward_supervision_index]
            mean_z = (selected_means * masks.squeeze(-1)).sum() / masks.sum().clamp(min=1.0)
            self.upward_supervision_metrics = {
                "Loss/upward_action_supervision": mean_supervision_loss / num_updates,
                "Aux/upward_action_supervision_coef": supervision_coef,
                "Aux/upward_action_supervision_fraction": masks.mean().item(),
                "Aux/upward_action_supervision_mean_z": mean_z.item(),
                "Aux/upward_action_supervision_decay_triggered": float(self.upward_supervision_decay_start is not None),
                "Aux/upward_action_supervision_lift_success_rate": (
                    statistics.mean(self.upward_supervision_lift_buffer)
                    if self.upward_supervision_lift_buffer else 0.0
                ),
            }

        return mean_value_loss, mean_surrogate_loss

    def adjust_learning_rate_cos(self, optimizer, epoch, max_epoch, iter, max_iter):
        lr = self.step_size_init * 0.5 * (1. + math.cos(math.pi * (iter + epoch / max_epoch) / max_iter))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    # def eval(self, maxlen, traj_index, traj_saved_path):
    def eval(self, logger, max_trajs=1000, maxlen=10e8, record_video=False):  # szn gedit
        # maxlen = 200

        cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

        reward_sum = []
        episode_length = []
        successes = []


        current_obs_state = self.vec_env.reset()
        recurrent_hidden_states = self.initial_recurrent_state()
        while len(reward_sum) < self.vec_env.num_envs*max_trajs:#*self.cfg_env["episodeLength"]:
            with torch.no_grad():
                # Compute the action
                # actions, actions_log_prob, values, mu, sigma, current_state, current_obs_feats = self.actor_critic.act(current_obs_state)
                actions, next_recurrent_hidden_states = (
                    self.act_inference(
                        current_obs_state,
                        recurrent_hidden_states,
                    )
                )
                # Step the vec_environment
                next_obs_state, rews, dones, infos = self.vec_env.step(actions)
                # next_obs_state, rews, dones, infos = self.vec_env.step(actions)
                # time.sleep(0.01)
                # self.vec_env.render()
                current_obs_state.copy_(next_obs_state)
                recurrent_hidden_states = self.mask_recurrent_state(
                    next_recurrent_hidden_states,
                    dones,
                )
                cur_reward_sum[:] += rews
                cur_episode_length[:] += 1

                new_ids = (dones > 0).nonzero(as_tuple=False)
                reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                successes.extend(infos["successes"][new_ids][:, 0].cpu().numpy().tolist())

                if len(new_ids) > 0:
                    # from utils.util import plot_tensor_data
                    # plot_tensor_data(self.vec_env.task.action_penalty, "vt-abs")
                    # plot_tensor_data(self.vec_env.task.action_penalty1, "vt-rela")
                    print("-" * 80)
                    print("Num episodes: {}".format(len(reward_sum)))
                    print("Mean return: {:.2f}".format(statistics.mean(reward_sum)))
                    print("Mean ep len: {:.2f}".format(statistics.mean(episode_length)))
                    print("Mean success: {:.2f}".format(statistics.mean(successes) * 100))
        logger.log_kv("Mean success", statistics.mean(successes) * 100)
