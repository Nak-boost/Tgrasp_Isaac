# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os.path as osp
from utils.torch_jit_utils import *
from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from dexrep.ShareDexRepSensor import SharedDexRepSensor as DexRepEncoder
_DexRepEncoder_Map = {
            'DexRep': DexRepEncoder,
            'DexRep_debug': DexRepEncoder,
        }

class ShadowHandGraspDexRep(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)
        self.obs_type = self.cfg["env"]["observationType"]
        print("Obs type:", self.obs_type)

        self.tactile_cfg = self.cfg["env"].get("tactile", {})
        self.tactile_enabled = self.tactile_cfg.get("enabled", False)
        self.tactile_experiment = self.tactile_cfg.get(
            "experiment", "E1"
        ).upper()

        self.fingertips = [
            "robot0:ffdistal",
            "robot0:mfdistal",
            "robot0:rfdistal",
            "robot0:lfdistal",
            "robot0:thdistal",
        ]
        touch_layout = self.tactile_cfg.get("touch", {}).get(
            "layout", "pad14" if self.tactile_enabled else "fingertip5"
        )
        touch_layouts = {
            "fingertip5": list(self.fingertips),
            "link14": [
                *self.fingertips,
                "robot0:ffmiddle",
                "robot0:mfmiddle",
                "robot0:rfmiddle",
                "robot0:lfmiddle",
                "robot0:ffproximal",
                "robot0:mfproximal",
                "robot0:rfproximal",
                "robot0:lfproximal",
                "robot0:palm",
            ],
            "pad14": [
                "robot0:ffdistal_sensor",
                "robot0:mfdistal_sensor",
                "robot0:rfdistal_sensor",
                "robot0:lfdistal_sensor",
                "robot0:thdistal_sensor",
                "robot0:ffmiddle_sensor",
                "robot0:mfmiddle_sensor",
                "robot0:rfmiddle_sensor",
                "robot0:lfmiddle_sensor",
                "robot0:ffproximal_sensor",
                "robot0:mfproximal_sensor",
                "robot0:rfproximal_sensor",
                "robot0:lfproximal_sensor",
                "robot0:palm_sensor",
            ],
        }
        if touch_layout not in touch_layouts:
            raise ValueError(
                f"Unknown tactile touch layout '{touch_layout}'. "
                f"Expected one of {sorted(touch_layouts)}"
            )
        self.touch_layout = touch_layout
        self.touch_sensor_bodies = touch_layouts[touch_layout]
        regular_finger_inward_normal = [0.0, 1.0, 0.0]
        thumb_inward_normal = [1.0, 0.0, 0.0]
        self.touch_sensor_local_inward_normals_values = (
            [regular_finger_inward_normal] * 4
            + [thumb_inward_normal]
            + [regular_finger_inward_normal] * 9
        )[:len(self.touch_sensor_bodies)]
        touch_cfg = self.tactile_cfg.get("touch", {})
        self.touch_force_mode = touch_cfg.get(
            "force_mode",
            "normal" if touch_layout == "pad14" else "norm",
        )
        self.touch_minimum_height_above_table = float(
            touch_cfg.get("minimum_height_above_table", 0.005)
        )
        if self.touch_force_mode not in {"normal", "norm"}:
            raise ValueError(
                "tactile.touch.force_mode must be normal or norm"
            )
        if self.touch_minimum_height_above_table < 0.0:
            raise ValueError(
                "tactile.touch.minimum_height_above_table must be "
                "non-negative"
            )
        self.num_fingertips = len(self.fingertips)
        self.num_touch_sensors = len(self.touch_sensor_bodies)
        finger_prefixes = (
            "robot0:ff",
            "robot0:mf",
            "robot0:rf",
            "robot0:lf",
            "robot0:th",
        )
        self.touch_sensor_finger_membership_values = [
            [
                body_name.startswith(finger_prefix)
                for body_name in self.touch_sensor_bodies
            ]
            for finger_prefix in finger_prefixes
        ]

        self.use_dexrep = False
        self.use_pnG = False
        self.use_geodex = False

        if self.tactile_enabled:
            valid_experiments = {"E1", "E2", "E3", "E4", "E5"}
            if self.tactile_experiment not in valid_experiments:
                raise ValueError(
                    f"Unknown tactile experiment "
                    f"'{self.tactile_experiment}'. "
                    f"Expected one of {sorted(valid_experiments)}"
                )

            self.tactile_hand_dof_dim = self.tactile_cfg.get(
                "hand_dof_count", 22
            )
            self.tactile_action_dim = 24
            self.current_touch_dim = self.num_touch_sensors
            self.relative_object_position_dim = 3
            self.oracle_object_dim = 10

            relative_position_cfg = self.tactile_cfg.get(
                "relative_object_position", {}
            )
            self.relative_object_position_enabled = bool(
                relative_position_cfg.get("enabled", True)
            )
            self.relative_object_position_source = str(
                relative_position_cfg.get("source", "oracle")
            ).lower()
            if self.relative_object_position_source not in {
                "oracle",
                "tactile_estimate",
            }:
                raise ValueError(
                    "tactile.relative_object_position.source must be "
                    "oracle or tactile_estimate"
                )
            self.relative_object_position_mask_probability = float(
                relative_position_cfg.get("mask_probability", 0.0)
            )
            if not (
                0.0
                <= self.relative_object_position_mask_probability
                <= 1.0
            ):
                raise ValueError(
                    "tactile.relative_object_position.mask_probability "
                    "must be between 0 and 1"
                )
            relative_position_scale = np.asarray(
                relative_position_cfg.get(
                    "scale", [0.35, 0.35, 0.40]
                ),
                dtype=np.float32,
            )
            if (
                relative_position_scale.shape != (3,)
                or np.any(relative_position_scale <= 0.0)
            ):
                raise ValueError(
                    "tactile.relative_object_position.scale must "
                    "contain three positive values"
                )
            self.relative_object_position_scale_values = (
                relative_position_scale.tolist()
            )
            estimator_cfg = relative_position_cfg.get("estimator", {})
            self.tactile_position_reliable_alpha = float(
                estimator_cfg.get("reliable_alpha", 0.6)
            )
            self.tactile_position_fallback_alpha = float(
                estimator_cfg.get("fallback_alpha", 0.2)
            )
            self.tactile_position_fallback_quality = float(
                estimator_cfg.get("fallback_quality", 0.3)
            )
            self.tactile_position_quality_decay = float(
                estimator_cfg.get("quality_decay", 0.99)
            )
            self.tactile_position_minimum_quality = float(
                estimator_cfg.get("minimum_quality", 0.1)
            )
            self.tactile_position_minimum_contacts = int(
                estimator_cfg.get("minimum_contacts", 2)
            )
            self.tactile_position_minimum_eigenvalue_ratio = float(
                estimator_cfg.get("minimum_eigenvalue_ratio", 0.05)
            )
            self.tactile_position_maximum_residual = float(
                estimator_cfg.get("maximum_residual", 0.03)
            )
            self.tactile_position_minimum_forward_fraction = float(
                estimator_cfg.get("minimum_forward_fraction", 0.6)
            )
            self.tactile_position_maximum_center_distance = float(
                estimator_cfg.get("maximum_center_distance", 0.20)
            )
            self.tactile_position_maximum_hand_distance = float(
                estimator_cfg.get("maximum_hand_distance", 0.40)
            )
            self.tactile_position_maximum_height_above_table = float(
                estimator_cfg.get("maximum_height_above_table", 0.40)
            )
            estimator_unit_interval_values = (
                self.tactile_position_reliable_alpha,
                self.tactile_position_fallback_alpha,
                self.tactile_position_fallback_quality,
                self.tactile_position_quality_decay,
                self.tactile_position_minimum_quality,
                self.tactile_position_minimum_forward_fraction,
            )
            if not all(
                0.0 <= value <= 1.0
                for value in estimator_unit_interval_values
            ):
                raise ValueError(
                    "tactile position estimator fractions must be in [0, 1]"
                )
            if self.tactile_position_minimum_contacts < 1:
                raise ValueError(
                    "tactile position minimum_contacts must be positive"
                )
            estimator_positive_values = (
                self.tactile_position_minimum_eigenvalue_ratio,
                self.tactile_position_maximum_residual,
                self.tactile_position_maximum_center_distance,
                self.tactile_position_maximum_hand_distance,
                self.tactile_position_maximum_height_above_table,
            )
            if not all(
                value > 0.0 for value in estimator_positive_values
            ):
                raise ValueError(
                    "tactile position estimator limits must be positive"
                )

            geometry_cfg = self.tactile_cfg.get(
                "oracle_geometry", {}
            )
            self.pointnet_global_dim = int(
                geometry_cfg.get("pointnet_global_dim", 1024)
            )
            self.dexrep_interaction_dim = int(
                geometry_cfg.get("dexrep_interaction_dim", 2360)
            )
            self.tactile_core_dim = (
                2 * self.tactile_hand_dof_dim + 6
            )
            self.tactile_prop_dim = (
                self.tactile_core_dim
                + 3 * self.num_fingertips
                + self.tactile_action_dim
            )
            self.history_frame_dim = (
                4 * self.current_touch_dim
                + self.tactile_core_dim
                + self.tactile_action_dim
            )
            self.tactile_cfg["history"][
                "frame_dim"
            ] = self.history_frame_dim

            obs_dim = {
                "prop": self.tactile_prop_dim,
                "current_touch": self.current_touch_dim,
                "relative_object_position": (
                    self.relative_object_position_dim
                ),
            }

            if self.tactile_experiment in {"E2", "E5"}:
                obs_dim["oracle_object"] = self.oracle_object_dim
                obs_dim[
                    "pointnet_global"
                ] = self.pointnet_global_dim
                if self.tactile_experiment == "E5":
                    obs_dim[
                        "dexrep_interaction"
                    ] = self.dexrep_interaction_dim
            elif self.tactile_experiment == "E3":
                voxel_cfg = self.tactile_cfg["voxel"]
                voxel_size = int(np.prod(voxel_cfg["grid_size"]))
                obs_dim["voxel_map"] = (
                    voxel_cfg.get("channels", 2) * voxel_size
                )
                local_cfg = self.tactile_cfg["local_voxel"]
                obs_dim["local_voxel_map"] = (
                    local_cfg["channels"]
                    * int(np.prod(local_cfg["grid_size"]))
                )
            elif self.tactile_experiment == "E4":
                history_length = self.tactile_cfg["history"]["length"]
                obs_dim["touch_history"] = (
                    history_length * self.history_frame_dim
                )

            self.cfg["env"]["obs_dim"] = obs_dim
            self.cfg["env"]["numObservations"] = sum(obs_dim.values())

        num_obs = 236 + 64
        self.num_obs_dict = {
            "full_state": num_obs,
            "DexRep": 2567
        }
        # if use DexRep Encoder
        if (
            self.tactile_enabled
            and self.tactile_experiment in {"E2", "E5"}
        ):
            if "dexrep" not in cfg:
                raise ValueError(
                    "E2 and E5 require the dexrep configuration"
                )
            self.use_dexrep = True
            self.DexRepEncoder = DexRepEncoder(
                cfg, device_type + f":{device_id}"
            )
        elif self.tactile_enabled:
            self.use_dexrep = False
        elif self.obs_type in _DexRepEncoder_Map.keys():
            assert "dexrep" in cfg.keys()
            self.use_dexrep = True
            self.DexRepEncoder = _DexRepEncoder_Map[self.obs_type](cfg, device_type+f":{device_id}")

        self.num_hand_obs = 66 + 95 + 24 + 6  # 191 =  22*3 + (65+30) + 24
        self.up_axis = 'z'
        self.hand_center = ["robot0:palm"]
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        num_states = 0
        if self.asymmetric_obs:
            num_states = 211
        if not self.tactile_enabled:
            self.cfg["env"]["numObservations"] = self.num_obs_dict[
                self.obs_type
            ]
        self.cfg["env"]["numStates"] = num_states
        self.num_agents = 1
        self.cfg["env"]["numActions"] = 24 
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self.dexrep_hand = [
            "robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal",
            "robot0:ffmiddle", "robot0:mfmiddle", "robot0:rfmiddle", "robot0:lfmiddle", "robot0:thmiddle",
            "robot0:ffproximal", "robot0:mfproximal", "robot0:rfproximal", "robot0:lfmetacarpal", "robot0:thproximal"
        ]

        # self.dexrep_hand_handles = [self.gym.find_asset_rigid_body_index(self.shadow_hand_asset, name) for name in
        #                             self.dexrep_hand]
        super().__init__(cfg=self.cfg, enable_camera_sensors=True)

        self.num_dexrep_hand = len(self.dexrep_hand)
        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # Net contact forces contain rigid-body collision forces without
        # articulation drive and internal joint-constraint forces.
        net_contact_force_tensor = (
            self.gym.acquire_net_contact_force_tensor(self.sim)
        )
        if not self.tactile_enabled:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(
                self.sim
            )
            self.vec_sensor_tensor = gymtorch.wrap_tensor(
                sensor_tensor
            ).view(self.num_envs, self.num_touch_sensors * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,
                                                self.num_shadow_hand_dofs + self.num_object_dofs)
        self.dof_force_tensor = self.dof_force_tensor[:, :self.num_shadow_hand_dofs]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.net_contact_force_tensor = gymtorch.wrap_tensor(
            net_contact_force_tensor
        ).view(self.num_envs, self.num_bodies, 3)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,-1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        if self.tactile_enabled:
            self.relative_object_position_scale = to_torch(
                self.relative_object_position_scale_values,
                device=self.device,
                dtype=torch.float,
            ).view(1, 3)
            self.relative_object_position_visibility = torch.ones(
                self.num_envs,
                1,
                device=self.device,
                dtype=torch.float,
            )
            self.touch_sensor_local_inward_normals = to_torch(
                self.touch_sensor_local_inward_normals_values,
                device=self.device,
                dtype=torch.float,
            )
            self.touch_sensor_world_inward_normals = torch.zeros(
                self.num_envs,
                self.num_touch_sensors,
                3,
                device=self.device,
                dtype=torch.float,
            )
            self.tactile_object_position_estimate = torch.zeros(
                self.num_envs,
                3,
                device=self.device,
                dtype=torch.float,
            )
            self.tactile_object_position_quality = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            )
            self.tactile_position_previous_contact_positions = torch.zeros(
                self.num_envs,
                self.num_touch_sensors,
                3,
                device=self.device,
                dtype=torch.float,
            )
            self.tactile_position_previous_contact_directions = torch.zeros(
                self.num_envs,
                self.num_touch_sensors,
                3,
                device=self.device,
                dtype=torch.float,
            )
            self.tactile_position_previous_touch = torch.zeros(
                self.num_envs,
                self.num_touch_sensors,
                device=self.device,
                dtype=torch.bool,
            )

            if (
                self.num_shadow_hand_dofs
                != self.tactile_hand_dof_dim
            ):
                raise ValueError(
                    "Configured tactile.hand_dof_count is "
                    f"{self.tactile_hand_dof_dim}, but the loaded "
                    f"hand has {self.num_shadow_hand_dofs} DOFs"
                )

            self.binary_touch = torch.zeros(
                self.num_envs,
                self.num_touch_sensors,
                device=self.device,
                dtype=torch.float,
            )
            self.touch_sensor_finger_membership = torch.tensor(
                self.touch_sensor_finger_membership_values,
                device=self.device,
                dtype=torch.bool,
            )
            self.previous_binary_touch = torch.zeros_like(
                self.binary_touch
            )
            self.consecutive_valid_grasp_steps = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.long,
            )
            self.valid_grasp_mask = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.bool,
            )

            touch_reward_cfg = self.tactile_cfg.get("reward", {})
            self.touch_multi_contact_hold_reward_scale = (
                touch_reward_cfg.get("multi_contact_hold", 0.02)
            )
            self.touch_contact_per_finger_reward_scale = float(
                touch_reward_cfg.get("per_finger_contact", 0.01)
            )
            self.touch_valid_grasp_contact_reward_scale = float(
                touch_reward_cfg.get("valid_grasp_contact", 0.02)
            )
            self.estimated_reach_reward_scale = float(
                touch_reward_cfg.get("estimated_reach", 0.0)
            )
            self.valid_grasp_required_steps = int(
                touch_reward_cfg.get("valid_grasp_steps", 3)
            )
            self.touch_upward_action_reward_scale = float(
                touch_reward_cfg.get("upward_action", 0.075)
            )
            self.touch_upward_action_reward_saturation = float(
                touch_reward_cfg.get("upward_action_saturation", 0.3)
            )
            self.touch_lift_height_reward_scale = touch_reward_cfg.get(
                "lift_height", 0.25
            )
            self.touch_lift_target_height = touch_reward_cfg.get(
                "lift_target_height", 0.12
            )
            self.touch_lift_hold_reward_scale = touch_reward_cfg.get(
                "lift_hold", 0.5
            )
            self.touch_lift_hold_steps = int(
                touch_reward_cfg.get("lift_hold_steps", 10)
            )
            self.reach_palm_weight = touch_reward_cfg.get(
                "reach_palm_weight", 0.5
            )
            self.reach_distal_weight = touch_reward_cfg.get(
                "reach_distal_weight", 1.0
            )
            self.reach_middle_weight = touch_reward_cfg.get(
                "reach_middle_weight", 0.25
            )
            self.reach_proximal_weight = touch_reward_cfg.get(
                "reach_proximal_weight", 0.0
            )
            self.voxel_exploration_reward_scale = float(
                touch_reward_cfg.get("voxel_exploration", 0.01)
            )
            self.voxel_exploration_new_voxel_cap = float(
                touch_reward_cfg.get(
                    "voxel_exploration_new_voxel_cap", 4
                )
            )
            self.voxel_exploration_max_height_above_table = float(
                touch_reward_cfg.get(
                    "voxel_exploration_max_height_above_table", 0.20
                )
            )
            if self.touch_lift_hold_steps < 1:
                raise ValueError(
                    "tactile.reward.lift_hold_steps must be positive"
                )
            if self.valid_grasp_required_steps < 1:
                raise ValueError(
                    "tactile.reward.valid_grasp_steps must be positive"
                )
            if self.touch_upward_action_reward_saturation <= 0.0:
                raise ValueError(
                    "tactile.reward.upward_action_saturation must be positive"
                )
            if self.voxel_exploration_reward_scale < 0.0:
                raise ValueError(
                    "tactile.reward.voxel_exploration must be non-negative"
                )
            if self.voxel_exploration_new_voxel_cap <= 0.0:
                raise ValueError(
                    "tactile.reward.voxel_exploration_new_voxel_cap "
                    "must be positive"
                )
            if (
                self.voxel_exploration_max_height_above_table
                <= self.touch_minimum_height_above_table
            ):
                raise ValueError(
                    "tactile.reward.voxel_exploration_max_height_above_table "
                    "must exceed tactile.touch.minimum_height_above_table"
                )
            if (
                self.touch_contact_per_finger_reward_scale < 0.0
                or self.touch_valid_grasp_contact_reward_scale < 0.0
                or self.estimated_reach_reward_scale < 0.0
            ):
                raise ValueError("tactile contact and reach rewards must be non-negative")
            if (
                self.tactile_experiment == "E3"
                and self.estimated_reach_reward_scale > 0.0
                and self.relative_object_position_source != "tactile_estimate"
            ):
                raise ValueError("E3 estimated reach requires tactile_estimate")

            self.episode_had_contact = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.bool,
            )
            self.episode_had_valid_estimate = torch.zeros_like(
                self.episode_had_contact
            )
            self.episode_had_multi_contact = torch.zeros_like(
                self.episode_had_contact
            )
            self.episode_had_lift = torch.zeros_like(
                self.episode_had_contact
            )
            self.episode_first_contact_step = torch.full(
                (self.num_envs,),
                -1.0,
                device=self.device,
            )
            self.episode_contact_steps = torch.zeros(
                self.num_envs,
                device=self.device,
            )
            self.episode_elapsed_steps = torch.zeros_like(
                self.episode_contact_steps
            )
            self.episode_contact_losses = torch.zeros_like(
                self.episode_contact_steps
            )
            self.voxel_exploration_reward = torch.zeros_like(
                self.episode_contact_steps
            )
            self.episode_object_start_height = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            )
            self.previous_reach_distance = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            )
            self.previous_reach_distance_valid = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.bool,
            )
            self.previous_reach_keypoints = torch.zeros(
                self.num_envs, 16, 3, device=self.device,
            )
            self.consecutive_lift_hold_steps = torch.zeros_like(
                self.episode_contact_steps
            )

            history_length = self.tactile_cfg["history"]["length"]
            self.touch_history = torch.zeros(
                self.num_envs,
                history_length,
                self.history_frame_dim,
                device=self.device,
                dtype=torch.float,
            )

            voxel_cfg = self.tactile_cfg["voxel"]
            self.voxel_grid_size = tuple(voxel_cfg["grid_size"])
            self.voxel_grid_size_tensor = to_torch(
                self.voxel_grid_size,
                device=self.device,
                dtype=torch.long,
            )
            self.voxel_channels = voxel_cfg.get("channels", 2)
            self.voxel_lower = to_torch(
                voxel_cfg["lower"], device=self.device
            )
            self.voxel_upper = to_torch(
                voxel_cfg["upper"], device=self.device
            )
            self.voxel_map = torch.zeros(
                self.num_envs,
                self.voxel_channels,
                *self.voxel_grid_size,
                device=self.device,
                dtype=torch.float,
            )
            if self.voxel_channels != 2:
                raise ValueError(
                    "The tactile voxel map requires two channels: "
                    "occupancy and recency"
                )
            self.voxel_map[:, 0].fill_(-1.0)
            voxel_height = (
                self.voxel_upper[2] - self.voxel_lower[2]
            ) / self.voxel_grid_size[2]
            voxel_lower_edges = self.voxel_lower[2] + torch.arange(
                self.voxel_grid_size[2], device=self.device
            ) * voxel_height
            reward_lower_z = (
                self.table_top_z + self.touch_minimum_height_above_table
            )
            reward_upper_z = (
                self.table_top_z
                + self.voxel_exploration_max_height_above_table
            )
            eligible_z = (
                (voxel_lower_edges >= reward_lower_z)
                & (voxel_lower_edges + voxel_height <= reward_upper_z)
            )
            self.voxel_exploration_z_mask = eligible_z.view(1, 1, 1, -1).expand(
                1, *self.voxel_grid_size
            ).flatten(1)
            self.voxel_recency_decay = float(
                voxel_cfg.get("recency_decay", 0.98)
            )
            self.voxel_expiration_threshold = float(
                voxel_cfg.get("expiration_threshold", 0.1)
            )
            if not 0.0 <= self.voxel_recency_decay <= 1.0:
                raise ValueError(
                    "tactile.voxel.recency_decay must be in [0, 1]"
                )
            if not 0.0 <= self.voxel_expiration_threshold <= 1.0:
                raise ValueError(
                    "tactile.voxel.expiration_threshold must be in [0, 1]"
                )
            if self.tactile_experiment == "E3":
                local_cfg = self.tactile_cfg["local_voxel"]
                self.local_voxel_grid_size = tuple(local_cfg["grid_size"])
                if (
                    len(self.local_voxel_grid_size) != 3
                    or any(size <= 0 or size % 2 for size in self.local_voxel_grid_size)
                    or local_cfg["channels"] != 8
                ):
                    raise ValueError(
                        "E3 local_voxel requires three positive even grid sizes "
                        "and eight channels"
                    )
                local_axes = [
                    torch.arange(size, device=self.device) - size // 2
                    for size in self.local_voxel_grid_size
                ]
                self.local_voxel_offsets = torch.stack(
                    torch.meshgrid(*local_axes, indexing="ij"), dim=-1
                ).reshape(1, -1, 3)
                self.local_voxel_half_size = torch.tensor(
                    self.local_voxel_grid_size,
                    device=self.device,
                    dtype=torch.long,
                ).div(2, rounding_mode="floor")
                self.local_voxel_grid_size_tensor = (
                    self.local_voxel_half_size * 2
                )

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # TODO: using grab objects
        object_scale_dict = self.cfg['env']['object_code_dict']
        self.object_code_list = object_scale_dict

        assets_path = '../assets'
        print(f'Num Objs: {len(self.object_code_list)}')
        print(f'Num Envs: {self.num_envs}')

        self.goal_cond = self.cfg["env"]["goal_cond"]
        self.random_time = self.cfg["env"]["random_time"]
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        shadow_hand_asset, shadow_hand_dof_props, table_texture_handle = self._load_shadow_hand_asset()

        goal_asset_dict, object_asset_dict = self._load_object_asset(assets_path)

        # create table asset
        table_asset, table_dims = self._load_table_asset()

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.05, 0.8)  # gymapi.Vec3(0.1, 0.1, 0.65)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.6 + 0.1)  # gymapi.Vec3(0.0, 0.0, 0.72)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.2)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)

        goal_start_pose.p.z -= 0.0

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)
        self.table_top_z = table_pose.p.z + 0.5 * table_dims.z

        # compute aggregate size
        # max_agg_bodies = self.num_shadow_hand_bodies * 1 + 2 * self.num_object_bodies + 1  ##
        # max_agg_shapes = self.num_shadow_hand_shapes * 1 + 2 * self.num_object_shapes + 1  ##

        self.shadow_hands = []
        self.objects = []
        self.envs = []
        self.object_init_state = []
        self.goal_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []
        self.dexrep_hand_indices = []
        for o in range(len(self.dexrep_hand)):
            dexrep_hand_env_handle = self.gym.find_asset_rigid_body_index(shadow_hand_asset, self.dexrep_hand[o])
            self.dexrep_hand_indices.append(dexrep_hand_env_handle)
        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(
                shadow_hand_asset, name
            )
            for name in self.fingertips
        ]
        self.touch_sensor_handles = [
            self.gym.find_asset_rigid_body_index(
                shadow_hand_asset, name
            )
            for name in self.touch_sensor_bodies
        ]
        missing_touch_bodies = [
            body_name
            for body_name, body_handle in zip(
                self.touch_sensor_bodies,
                self.touch_sensor_handles,
            )
            if body_handle < 0
        ]
        if missing_touch_bodies:
            raise RuntimeError(
                "Touch sensor bodies are missing from the Shadow Hand "
                f"asset: {missing_touch_bodies}"
            )

        body_names = {
            'wrist': 'robot0:wrist',
            'palm': 'robot0:palm',
            'thumb': 'robot0:thdistal',
            'index': 'robot0:ffdistal',
            'middle': 'robot0:mfdistal',
            'ring': 'robot0:rfdistal',
            'little': 'robot0:lfdistal'
        }
        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(shadow_hand_asset, body_name)

        # Preserve the original wrench observations used by the
        # non-tactile DexRep baseline. Tactile experiments instead use
        # rigid-body net contact forces.
        if not self.tactile_enabled:
            sensor_pose = gymapi.Transform()
            for touch_sensor_handle in self.touch_sensor_handles:
                self.gym.create_asset_force_sensor(
                    shadow_hand_asset,
                    touch_sensor_handle,
                    sensor_pose,
                )

        # self.object_scale_buf = {}

        for i in range(self.num_envs):
            object_idx_this_env = i % len(self.object_code_list)
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            max_agg_bodies = self.num_shadow_hand_bodies + self.num_object_bodies_list[object_idx_this_env] + 2
            max_agg_shapes = self.num_shadow_hand_shapes + self.num_object_shapes_list[object_idx_this_env] + 2

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # load shadow hand  for each env
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append(
                [shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                 shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z,
                 shadow_hand_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])

            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)


            # Color by resolved body type rather than legacy rigid-body
            # indices, which change when tactile pad bodies are preserved.
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, shadow_hand_actor)
            hand_color = gymapi.Vec3(147/255, 215/255, 160/255)
            for body_index in range(num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr,
                    shadow_hand_actor,
                    body_index,
                    gymapi.MESH_VISUAL,
                    hand_color,
                )
            if self.tactile_enabled:
                sensor_color = gymapi.Vec3(1.0, 1.0, 1.0)
                for touch_sensor_handle in self.touch_sensor_handles:
                    self.gym.set_rigid_body_color(
                        env_ptr,
                        shadow_hand_actor,
                        touch_sensor_handle,
                        gymapi.MESH_VISUAL_AND_COLLISION,
                        sensor_color,
                    )
            # create fingertip force-torque sensors
            # if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            # load object for each env

            object_handle = self.gym.create_actor(env_ptr, object_asset_dict[object_idx_this_env], object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                         goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z,
                                         goal_start_pose.r.w,
                                         0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.gym.set_actor_scale(env_ptr, object_handle, 1)
            # DexRep or pnG load object
            if self.use_dexrep:
                self.DexRepEncoder.load_batch_env_obj(object_idx_this_env)
            elif self.use_pnG:
                self.PnGEncoder.load_batch_env_obj(object_idx_this_env)
            elif self.use_geodex:
                self.GeoDexWrapper.load_batch_env_obj(object_idx_this_env)
            # add goal object
            # goal_asset_dict[id][scale_id]
            goal_handle = self.gym.create_actor(env_ptr, goal_asset_dict[object_idx_this_env], goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_scale(env_ptr, goal_handle, 1.0)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            # set friction
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            table_shape_props[0].friction = 1
            object_shape_props[0].friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

            object_color = [90/255, 94/255, 173/255]
            self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color))
            table_color = [150/255, 150/255, 150/255]
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*table_color))
            
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)
            self.objects.append(object_handle)


        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.goal_init_state.clone()
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        self.goal_states[:, self.up_axis_idx] -= 0.04

        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.touch_sensor_handles = to_torch(
            self.touch_sensor_handles,
            dtype=torch.long,
            device=self.device,
        )
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)

    def _load_table_asset(self):
        table_dims = gymapi.Vec3(1, 1, 0.6)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions())
        return table_asset, table_dims

    def _load_object_asset(self, assets_path):
        object_asset_dict = {}
        goal_asset_dict = {}
        self.num_object_bodies_list = []
        self.num_object_shapes_list = []
        # mesh_path = osp.join(assets_path, 'meshdatav3_scaled')
        self.asset_root = self.cfg["env"]["asset"]["assetRoot"]
        self.obj_asset_root = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj"]
        self.raw_obj_asset_root = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj_raw"]
        for object_id, object_code in enumerate(self.object_code_list):
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 500
            object_asset_options.fix_base_link = False
            # object_asset_options.disable_gravity = True
            object_asset_options.use_mesh_materials = True
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 300000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            object_asset = None
            object_asset_file = "coacd.urdf"
            object_asset = self.gym.load_asset(self.sim, self.obj_asset_root + f'grab-{object_code}' + "/coacd", object_asset_file, object_asset_options)
            if object_asset is None:
                print(object_code)
            assert object_asset is not None

            object_asset_options.disable_gravity = True
            goal_asset = self.gym.create_sphere(self.sim, 0.005, object_asset_options)

            dexrep_load = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj_raw"] + f'grab-{object_code}.obj'
            if self.use_dexrep:
                self.DexRepEncoder.load_cache_stl_file(
                    obj_idx=object_id,
                    obj_path=dexrep_load,
                    scale=1)
            elif self.use_pnG:
                self.PnGEncoder.load_cache_stl_file(
                    obj_idx=object_id,
                    obj_path=dexrep_load,
                    scale=1
                )
            elif self.use_geodex:
                self.GeoDexWrapper.load_cache_stl_file(
                    obj_idx=object_id,
                    obj_path=dexrep_load,
                    scale=1
                )
            # self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
            # self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
            self.num_object_bodies_list.append(self.gym.get_asset_rigid_body_count(object_asset))
            self.num_object_shapes_list.append(self.gym.get_asset_rigid_shape_count(object_asset))
            # set object dof properties
            self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset)
            self.object_dof_lower_limits = []
            self.object_dof_upper_limits = []

            for i in range(self.num_object_dofs):
                self.object_dof_lower_limits.append(object_dof_props['lower'][i])
                self.object_dof_upper_limits.append(object_dof_props['upper'][i])

            self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
            self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)
            object_asset_dict[object_id] = object_asset
            goal_asset_dict[object_id] = goal_asset
        return goal_asset_dict, object_asset_dict

    def _load_shadow_hand_asset(self):
        asset_root = "../../assets"
        shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)
        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file)
        if self.tactile_enabled:
            shadow_hand_asset_file = self.tactile_cfg.get(
                "touch", {}
            ).get(
                "asset_file",
                "mjcf/open_ai_assets/hand/shadow_hand_tactile.xml",
            )
        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = not self.tactile_enabled
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)
        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)
        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)
        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in
                              range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in
                                     actuated_dof_names]
        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)
        return shadow_hand_asset, shadow_hand_dof_props, table_texture_handle

    def compute_reward(self, actions, id=-1):
        self.dof_pos = self.shadow_hand_dof_pos

        if self.tactile_enabled:
            valid_grasp = self.update_valid_grasp()
            touch_count = valid_grasp.float() * 2.0
            object_start_height = self.episode_object_start_height
            lift_target_height = self.touch_lift_target_height
            lift_hold_reward_scale = (
                self.touch_lift_hold_reward_scale
            )
            lift_hold_steps_required = float(
                self.touch_lift_hold_steps
            )
            above_lift_target = (
                (
                    self.object_pos[:, 2] - object_start_height
                    >= lift_target_height
                )
                & (touch_count >= 2.0)
            )
            self.consecutive_lift_hold_steps.copy_(
                torch.where(
                    above_lift_target,
                    self.consecutive_lift_hold_steps + 1.0,
                    torch.zeros_like(self.consecutive_lift_hold_steps),
                )
            )
            lift_hold_steps = self.consecutive_lift_hold_steps
        else:
            touch_count = torch.zeros_like(self.object_pos[:, 2])
            object_start_height = self.object_pos[:, 2]
            lift_target_height = 0.12
            lift_hold_reward_scale = 0.0
            lift_hold_steps_required = 1.0
            lift_hold_steps = torch.zeros_like(touch_count)

        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.object_init_z, object_start_height, touch_count,
            lift_target_height, lift_hold_reward_scale,
            lift_hold_steps, lift_hold_steps_required,
            self.id, self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
            self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
            self.goal_pos, self.goal_rot,
            self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
            self.right_hand_lf_pos, self.right_hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor,
            self.goal_cond, self.tactile_enabled
        )

        if self.tactile_enabled:
            self.rew_buf.add_(self.compute_reach_progress_reward())
            self.rew_buf.add_(self.compute_touch_reward())
            if self.tactile_experiment == "E3":
                self.rew_buf.add_(self.voxel_exploration_reward)

        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes
        if self.tactile_enabled:
            self.extras["contact_found"] = (
                self.episode_had_contact.float()
            )
            if self.tactile_experiment == "E3":
                self.extras["estimate_found"] = (
                    self.episode_had_valid_estimate.float()
                )
            self.extras["first_contact_step"] = (
                self.episode_first_contact_step
            )
            self.extras["multi_contact"] = (
                self.episode_had_multi_contact.float()
            )
            self.extras["lift_hold_steps"] = (
                self.consecutive_lift_hold_steps
            )
            self.extras["lifted"] = self.episode_had_lift.float()
            self.extras["contact_step_ratio"] = (
                self.episode_contact_steps
                / torch.clamp(self.episode_elapsed_steps, min=1.0)
            )
            self.extras["contact_losses"] = (
                self.episode_contact_losses
            )
            self.extras["current_contact"] = (
                self.binary_touch > 0.5
            ).any(dim=1).float()
            self.extras["valid_grasp"] = (
                self.valid_grasp_mask.float()
            )
            self.extras["upward_action_supervision_mask"] = (
                self.valid_grasp_mask
                & (
                    self.object_pos[:, 2] - self.episode_object_start_height
                    < self.touch_lift_target_height
                )
                & (self.reset_buf == 0)
            ).float()

    def reach_distance_to_target(self, keypoints, target):
        distances = torch.norm(
            keypoints - target.unsqueeze(1), p=2, dim=2,
        )
        return (
            self.reach_palm_weight * distances[:, 0]
            + self.reach_distal_weight * distances[:, 1:6].mean(dim=1)
            + self.reach_middle_weight * distances[:, 6:11].mean(dim=1)
            + self.reach_proximal_weight * distances[:, 11:16].mean(dim=1)
        )

    def compute_reach_progress_reward(self):
        distal_keypoints = torch.stack(
            (
                self.right_hand_ff_pos,
                self.right_hand_mf_pos,
                self.right_hand_rf_pos,
                self.right_hand_lf_pos,
                self.right_hand_th_pos,
            ),
            dim=1,
        )
        middle_keypoints = self.dexrep_hand_state[:, 5:10, 0:3]
        proximal_keypoints = self.dexrep_hand_state[:, 10:15, 0:3]
        keypoints = torch.cat(
            (
                self.right_hand_pos.unsqueeze(1),
                distal_keypoints,
                middle_keypoints,
                proximal_keypoints,
            ),
            dim=1,
        )
        reach_distance = self.reach_distance_to_target(
            keypoints, self.object_handle_pos,
        )
        reach_progress = torch.where(
            self.previous_reach_distance_valid,
            self.previous_reach_distance - reach_distance,
            torch.zeros_like(reach_distance),
        )
        reward = self.dist_reward_scale * reach_progress
        if self.tactile_experiment == "E3":
            estimate = self.tactile_object_position_estimate
            estimated_progress = (
                self.reach_distance_to_target(
                    self.previous_reach_keypoints, estimate,
                )
                - self.reach_distance_to_target(keypoints, estimate)
            )
            estimated_progress = torch.where(
                self.previous_reach_distance_valid
                & (self.tactile_object_position_quality
                   >= self.tactile_position_minimum_quality),
                estimated_progress,
                torch.zeros_like(estimated_progress),
            )
            reward = reward + self.estimated_reach_reward_scale * estimated_progress
        self.previous_reach_distance.copy_(reach_distance)
        self.previous_reach_keypoints.copy_(keypoints)
        self.previous_reach_distance_valid[:] = True
        return reward

    def aggregate_finger_touch(self, touch):
        touch_active = touch > 0.5
        return torch.logical_and(
            touch_active.unsqueeze(1),
            self.touch_sensor_finger_membership.unsqueeze(0),
        ).any(dim=2).float()

    def update_valid_grasp(self):
        finger_touch = self.aggregate_finger_touch(
            self.binary_touch
        ) > 0.5
        thumb_touch = finger_touch[:, 4]
        other_finger_touch = finger_touch[:, :4].any(dim=1)
        grasp_candidate = torch.logical_and(
            thumb_touch,
            other_finger_touch,
        )
        self.consecutive_valid_grasp_steps.copy_(
            torch.where(
                grasp_candidate,
                self.consecutive_valid_grasp_steps + 1,
                torch.zeros_like(self.consecutive_valid_grasp_steps),
            )
        )
        self.valid_grasp_mask.copy_(
            self.consecutive_valid_grasp_steps
            >= self.valid_grasp_required_steps
        )
        return self.valid_grasp_mask

    def compute_touch_reward(self):
        minimum_lift_target = 1.0e-6
        current_touch = self.binary_touch > 0.5
        previous_touch = self.previous_binary_touch > 0.5
        current_finger_touch = self.aggregate_finger_touch(
            current_touch
        )
        previous_finger_touch = self.aggregate_finger_touch(
            previous_touch
        )

        touch_count = current_finger_touch.sum(dim=1)
        has_grasp_contact = self.valid_grasp_mask
        if self.tactile_experiment in {"E3", "E4"}:
            contact_reward = (
                self.touch_contact_per_finger_reward_scale
                * torch.clamp(touch_count, max=2.0)
                + self.touch_valid_grasp_contact_reward_scale
                * has_grasp_contact.float()
            )
        else:
            stable_finger_touch = torch.logical_and(
                current_finger_touch > 0.5,
                previous_finger_touch > 0.5,
            )
            contact_reward = (
                self.touch_multi_contact_hold_reward_scale
                * stable_finger_touch[:, 4].float()
                * stable_finger_touch[:, :4].sum(dim=1).float()
            )
        has_contact = current_touch.any(dim=1)
        lost_contacts = torch.logical_and(
            previous_finger_touch > 0.5,
            current_finger_touch < 0.5,
        ).sum(dim=1).float()

        lift_amount = torch.clamp(
            self.object_pos[:, 2]
            - self.episode_object_start_height,
            min=0.0,
        )
        lift_fraction = torch.clamp(
            lift_amount
            / max(self.touch_lift_target_height, minimum_lift_target),
            max=1.0,
        )

        current_episode_step = self.episode_elapsed_steps + 1.0
        first_contact = torch.logical_and(
            has_contact,
            torch.logical_not(self.episode_had_contact),
        )
        multi_contact = touch_count >= 2.0
        self.episode_first_contact_step = torch.where(
            first_contact,
            current_episode_step,
            self.episode_first_contact_step,
        )
        self.episode_had_contact.logical_or_(has_contact)
        self.episode_had_multi_contact.logical_or_(multi_contact)
        self.episode_had_lift.logical_or_(
            (lift_amount >= self.touch_lift_target_height)
            & has_grasp_contact
        )
        self.episode_contact_steps.add_(has_contact.float())
        self.episode_elapsed_steps.copy_(current_episode_step)
        self.episode_contact_losses.add_(lost_contacts)

        upward_action_fraction = torch.clamp(
            self.actions[:, 2]
            / self.touch_upward_action_reward_saturation,
            min=0.0,
            max=1.0,
        )
        below_lift_target = lift_amount < self.touch_lift_target_height
        upward_action_reward = (
            self.touch_upward_action_reward_scale
            * upward_action_fraction
            * has_grasp_contact.float()
            * below_lift_target.float()
        )
        touch_reward = (
            contact_reward
            + upward_action_reward
            + self.touch_lift_height_reward_scale
            * lift_fraction
            * has_grasp_contact.float()
        )

        self.previous_binary_touch.copy_(self.binary_touch)
        return touch_reward

    def compute_observations(self):
        # TODO:using dexrep
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # if self.obs_type == "full_state" or self.asymmetric_obs:
        if self.tactile_enabled:
            self.gym.refresh_net_contact_force_tensor(self.sim)
        else:
            self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_handle_pos = self.object_pos  ##+ quat_apply(self.object_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.06)
        self.object_back_pos = self.object_pos + quat_apply(self.object_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        idx = self.hand_body_idx_dict['palm']
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        if self.use_dexrep or self.use_pnG or self.tactile_enabled:
            self.dexrep_hand_state = self.rigid_body_states[:, self.dexrep_hand_indices, :].view(self.num_envs, -1, 13)
            self.dexrep_hand_pos = self.dexrep_hand_state[:, :, 0:3]
            self.dexrep_hand_vel = self.dexrep_hand_state[:, :, 7:13]
            # compute fingertip
            idx = 0
            self.right_hand_ff_pos, self.right_hand_ff_rot = self.dexrep_hand_state[:, idx, 0:3], self.dexrep_hand_state[:, idx, 3:7]
            self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)

            idx = 1
            self.right_hand_mf_pos, self.right_hand_mf_rot = self.dexrep_hand_state[:, idx, 0:3], self.dexrep_hand_state[:, idx, 3:7]
            self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)

            idx = 2
            self.right_hand_rf_pos = self.dexrep_hand_state[:, idx, 0:3]
            self.right_hand_rf_rot = self.dexrep_hand_state[:, idx, 3:7]
            self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)

            idx = 3
            self.right_hand_lf_pos = self.dexrep_hand_state[:, idx, 0:3]
            self.right_hand_lf_rot = self.dexrep_hand_state[:, idx, 3:7]
            self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)

            idx = 4
            self.right_hand_th_pos = self.dexrep_hand_state[:, idx, 0:3]
            self.right_hand_th_rot = self.dexrep_hand_state[:, idx, 3:7]
            self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)
            # concatenate
            fingertip_pos = torch.cat(
                (self.right_hand_ff_pos.unsqueeze(-2),
                 self.right_hand_mf_pos.unsqueeze(-2),
                 self.right_hand_rf_pos.unsqueeze(-2),
                 self.right_hand_lf_pos.unsqueeze(-2),
                 self.right_hand_th_pos.unsqueeze(-2)),
                dim=1
            )
            self.dexrep_hand_pos = torch.cat(    # expected [B, 20, 3]
                (fingertip_pos, self.dexrep_hand_pos),
                dim=1
            )

        # self.fingertip_state = self.rigid_body_states[self.fingertip_indices].view(self.num_envs, -1, 13)
        # self.fingertip_pos = self.fingertip_state[:, :, 0:3]
        # self.fingertip_ori = self.fingertip_state[:, :, 3:7]
        # self.fingertip_lin_vel = self.fingertip_state[:, :, 7:10]
        # self.fingertip_ang_vel = self.fingertip_state[:, :, 10:13]
        # self.fingertip_vel = self.fingertip_state[:, :, 7:13]
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        self.touch_sensor_state = self.rigid_body_states[
            :, self.touch_sensor_handles
        ][:, :, 0:13]
        self.touch_sensor_pos = self.touch_sensor_state[:, :, 0:3]

        if self.tactile_enabled:
            self.obs_buf = self.compute_tactile_observations()
        elif self.obs_type in ['DexRep']:
            assert self.use_dexrep
            base_state = self.compute_full_state()
            base_state = torch.clamp(
                base_state,
                -self.cfg["env"]["clip_observations"],
                self.cfg["env"]["clip_observations"],
            )
            dexrep_obs = self.DexRepEncoder.pre_observation(
                obj_pos=self.object_pos,
                obj_rot=self.object_rot,
                hand_pos=self.dexrep_hand_state[:, 11, 0:3].squeeze(dim=1),
                hand_rot=self.dexrep_hand_state[:, 11, 3:7].squeeze(dim=1),
                joints_sate=self.dexrep_hand_pos,
                clip_range=self.cfg["env"]["clip_observations"]
            )
            # dexrep_obs = torch.clamp(dexrep_obs, -self.cfg["env"]["clip_observations"],
            #                      self.cfg["env"]["clip_observations"])
            self.obs_buf = torch.cat(
                (base_state, dexrep_obs),
                dim=1
            )
        else:
            raise AttributeError(f'{self.obs_type} not include..')

    def compute_binary_touch(self):
        force_vectors = self.net_contact_force_tensor[
            :, self.touch_sensor_handles, :
        ]
        sensor_rotations = self.touch_sensor_state[:, :, 3:7]
        local_inward_normals = (
            self.touch_sensor_local_inward_normals
            .unsqueeze(0)
            .expand(self.num_envs, -1, -1)
        )
        self.touch_sensor_world_inward_normals = quat_apply(
            sensor_rotations.reshape(-1, 4),
            local_inward_normals.reshape(-1, 3),
        ).view(self.num_envs, self.num_touch_sensors, 3)
        if self.touch_force_mode == "normal":
            touch_force = torch.clamp(
                torch.sum(
                    force_vectors
                    * self.touch_sensor_world_inward_normals,
                    dim=-1,
                ),
                min=0.0,
            )
        else:
            touch_force = torch.norm(
                force_vectors, p=2, dim=-1
            )
        valid_touch_height = (
            self.touch_sensor_pos[:, :, 2]
            >= self.table_top_z
            + self.touch_minimum_height_above_table
        )
        touch_force = torch.where(
            valid_touch_height,
            touch_force,
            torch.zeros_like(touch_force),
        )
        self.touch_force = touch_force

        touch_on_threshold = self.tactile_cfg["touch"].get(
            "on_threshold", 1.0
        )
        touch_off_threshold = self.tactile_cfg["touch"].get(
            "off_threshold", 0.5
        )

        if touch_off_threshold > touch_on_threshold:
            raise ValueError(
                "touch.off_threshold must not exceed "
                "touch.on_threshold"
            )

        was_touching = self.binary_touch > 0.5
        remain_touching = touch_force >= touch_off_threshold
        begin_touching = touch_force >= touch_on_threshold

        self.binary_touch = torch.where(
            was_touching,
            remain_touching,
            begin_touching,
        ).float()
        return self.binary_touch

    def compute_tactile_proprioception(self):
        joint_positions = unscale(
            self.shadow_hand_dof_pos,
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )
        joint_velocities = (
            self.vel_obs_scale * self.shadow_hand_dof_vel
        )

        hand_position = self.right_hand_pos
        hand_orientation = get_euler_xyz(
            self.hand_orientations[self.hand_indices, :]
        )
        hand_euler = torch.stack(hand_orientation, dim=-1)

        fingertip_positions = self.fingertip_pos
        fingertip_positions_flat = fingertip_positions.reshape(
            self.num_envs, -1
        )

        previous_actions = getattr(
            self,
            "actions",
            torch.zeros(
                self.num_envs,
                self.num_actions,
                device=self.device,
                dtype=torch.float,
            ),
        ).clone()

        proprioception_core = torch.cat(
            (
                joint_positions,
                joint_velocities,
                hand_position,
                hand_euler,
            ),
            dim=1,
        )
        proprioception = torch.cat(
            (
                proprioception_core,
                fingertip_positions_flat,
                previous_actions,
            ),
            dim=1,
        )

        if proprioception.shape[1] != self.tactile_prop_dim:
            raise RuntimeError(
                f"Expected {self.tactile_prop_dim} proprioception "
                f"values, got {proprioception.shape[1]}"
            )

        return (
            proprioception,
            proprioception_core,
            fingertip_positions,
            previous_actions,
        )

    def solve_tactile_contact_rays(
        self,
        contact_positions,
        contact_directions,
        active_contacts,
        has_current_touch,
    ):
        weights = active_contacts.float()
        contact_count = weights.sum(dim=1)
        safe_contact_count = contact_count.clamp(min=1.0)
        identity = torch.eye(
            3,
            device=self.device,
            dtype=contact_positions.dtype,
        ).view(1, 1, 3, 3)
        projection = identity - (
            contact_directions.unsqueeze(-1)
            * contact_directions.unsqueeze(-2)
        )
        weighted_projection = projection * weights[:, :, None, None]
        system_matrix = weighted_projection.sum(dim=1)
        projected_positions = torch.matmul(
            projection,
            contact_positions.unsqueeze(-1),
        ).squeeze(-1)
        system_vector = (
            projected_positions * weights.unsqueeze(-1)
        ).sum(dim=1)

        eigenvalues = torch.linalg.eigvalsh(system_matrix)
        eigenvalue_ratio = eigenvalues[:, 0] / eigenvalues[:, -1].clamp(
            min=1e-6
        )
        regularization = 1e-6 * torch.eye(
            3,
            device=self.device,
            dtype=contact_positions.dtype,
        ).unsqueeze(0)
        ray_solution = torch.linalg.solve(
            system_matrix + regularization,
            system_vector.unsqueeze(-1),
        ).squeeze(-1)
        centroid = (
            contact_positions * weights.unsqueeze(-1)
        ).sum(dim=1) / safe_contact_count.unsqueeze(-1)

        ray_offsets = ray_solution.unsqueeze(1) - contact_positions
        perpendicular_offsets = torch.matmul(
            projection,
            ray_offsets.unsqueeze(-1),
        ).squeeze(-1)
        residual = (
            torch.linalg.vector_norm(perpendicular_offsets, dim=-1)
            * weights
        ).sum(dim=1) / safe_contact_count
        forward_fraction = (
            (
                (ray_offsets * contact_directions).sum(dim=-1)
                >= 0.0
            ).float()
            * weights
        ).sum(dim=1) / safe_contact_count
        center_distance = torch.linalg.vector_norm(
            ray_solution - centroid,
            dim=-1,
        )

        contacts_are_valid = (
            has_current_touch
            & torch.isfinite(centroid).all(dim=1)
            & (centroid[:, 2] >= self.table_top_z)
            & (
                centroid[:, 2]
                <= self.table_top_z
                + self.tactile_position_maximum_height_above_table
            )
            & (
                torch.linalg.vector_norm(
                    centroid - self.right_hand_pos,
                    dim=-1,
                )
                <= self.tactile_position_maximum_hand_distance
            )
        )
        ray_solution_is_in_workspace = (
            torch.isfinite(ray_solution).all(dim=1)
            & (ray_solution[:, 2] >= self.table_top_z)
            & (
                ray_solution[:, 2]
                <= self.table_top_z
                + self.tactile_position_maximum_height_above_table
            )
            & (
                torch.linalg.vector_norm(
                    ray_solution - self.right_hand_pos,
                    dim=-1,
                )
                <= self.tactile_position_maximum_hand_distance
            )
        )
        ray_solution_is_reliable = (
            contacts_are_valid
            & ray_solution_is_in_workspace
            & (contact_count >= self.tactile_position_minimum_contacts)
            & (
                eigenvalue_ratio
                >= self.tactile_position_minimum_eigenvalue_ratio
            )
            & (residual <= self.tactile_position_maximum_residual)
            & (
                forward_fraction
                >= self.tactile_position_minimum_forward_fraction
            )
            & (
                center_distance
                <= self.tactile_position_maximum_center_distance
            )
        )
        return (
            ray_solution,
            centroid,
            contacts_are_valid,
            ray_solution_is_reliable,
        )

    def update_tactile_object_position_estimate(self):
        current_touch = self.binary_touch > 0.5
        current_positions = self.touch_sensor_pos
        current_directions = -self.touch_sensor_world_inward_normals
        contact_positions = torch.cat(
            (
                self.tactile_position_previous_contact_positions,
                current_positions,
            ),
            dim=1,
        )
        contact_directions = torch.cat(
            (
                self.tactile_position_previous_contact_directions,
                current_directions,
            ),
            dim=1,
        )
        active_contacts = torch.cat(
            (
                self.tactile_position_previous_touch,
                current_touch,
            ),
            dim=1,
        )
        (
            ray_solution,
            centroid,
            contacts_are_valid,
            ray_solution_is_reliable,
        ) = self.solve_tactile_contact_rays(
            contact_positions,
            contact_directions,
            active_contacts,
            current_touch.any(dim=1),
        )

        measurement = torch.where(
            ray_solution_is_reliable.unsqueeze(-1),
            ray_solution,
            centroid,
        )
        measurement_quality = torch.where(
            ray_solution_is_reliable,
            torch.ones_like(self.tactile_object_position_quality),
            torch.full_like(
                self.tactile_object_position_quality,
                self.tactile_position_fallback_quality,
            ),
        )
        measurement_alpha = torch.where(
            ray_solution_is_reliable,
            torch.full_like(
                self.tactile_object_position_quality,
                self.tactile_position_reliable_alpha,
            ),
            torch.full_like(
                self.tactile_object_position_quality,
                self.tactile_position_fallback_alpha,
            ),
        )
        estimate_was_valid = (
            self.tactile_object_position_quality
            >= self.tactile_position_minimum_quality
        )
        smoothed_measurement = (
            (1.0 - measurement_alpha).unsqueeze(-1)
            * self.tactile_object_position_estimate
            + measurement_alpha.unsqueeze(-1) * measurement
        )
        updated_estimate = torch.where(
            estimate_was_valid.unsqueeze(-1),
            smoothed_measurement,
            measurement,
        )
        self.tactile_object_position_estimate.copy_(
            torch.where(
                contacts_are_valid.unsqueeze(-1),
                updated_estimate,
                self.tactile_object_position_estimate,
            )
        )

        smoothed_quality = (
            (1.0 - measurement_alpha)
            * self.tactile_object_position_quality
            + measurement_alpha * measurement_quality
        )
        decayed_quality = (
            self.tactile_object_position_quality
            * self.tactile_position_quality_decay
        )
        self.tactile_object_position_quality.copy_(
            torch.where(
                contacts_are_valid,
                torch.where(
                    estimate_was_valid,
                    smoothed_quality,
                    measurement_quality,
                ),
                decayed_quality,
            )
        )
        self.tactile_position_previous_contact_positions.copy_(
            current_positions
        )
        self.tactile_position_previous_contact_directions.copy_(
            current_directions
        )
        self.tactile_position_previous_touch.copy_(current_touch)

    def compute_relative_object_position_observation(self):
        if not self.relative_object_position_enabled:
            return torch.zeros_like(self.right_hand_pos)
        if self.relative_object_position_source == "tactile_estimate":
            estimate_is_valid = (
                self.tactile_object_position_quality
                >= self.tactile_position_minimum_quality
            ).unsqueeze(-1)
            relative_position = (
                self.tactile_object_position_estimate
                - self.right_hand_pos
            ) / self.relative_object_position_scale
            relative_position = torch.where(
                estimate_is_valid,
                relative_position,
                torch.zeros_like(relative_position),
            )
        else:
            relative_position = (
                self.object_pos - self.right_hand_pos
            ) / self.relative_object_position_scale
        return (
            relative_position
            * self.relative_object_position_visibility
        )

    def compute_oracle_object_observation(self):
        return torch.cat(
            (
                self.object_rot,
                self.object_linvel,
                self.vel_obs_scale * self.object_angvel,
            ),
            dim=1,
        )

    def update_touch_history(
        self,
        proprioception_core,
        touch_sensor_positions,
        previous_actions,
    ):
        history_frame = torch.cat(
            (
                touch_sensor_positions.reshape(self.num_envs, -1),
                self.binary_touch,
                proprioception_core,
                previous_actions,
            ),
            dim=1,
        )

        if history_frame.shape[1] != self.history_frame_dim:
            raise RuntimeError(
                f"Expected history frame dimension "
                f"{self.history_frame_dim}, got "
                f"{history_frame.shape[1]}"
            )

        self.touch_history[:, :-1] = self.touch_history[:, 1:].clone()
        self.touch_history[:, -1] = history_frame
        return self.touch_history.flatten(1)

    def mark_voxel_evidence(
        self,
        evidence_map,
        positions,
        position_valid,
    ):
        positions = positions.reshape(
            self.num_envs, -1, 3
        )
        position_valid = position_valid.reshape(
            self.num_envs, -1
        )

        normalized_positions = (
            positions - self.voxel_lower
        ) / (self.voxel_upper - self.voxel_lower)
        voxel_indices = torch.floor(
            normalized_positions
            * self.voxel_grid_size_tensor.float()
        ).long()

        inside_map = torch.logical_and(
            voxel_indices >= 0,
            voxel_indices
            < self.voxel_grid_size_tensor.view(1, 1, 3),
        ).all(dim=-1)
        valid = torch.logical_and(
            position_valid,
            inside_map,
        )

        linear_indices = (
            voxel_indices[..., 0]
            * self.voxel_grid_size[1]
            * self.voxel_grid_size[2]
            + voxel_indices[..., 1]
            * self.voxel_grid_size[2]
            + voxel_indices[..., 2]
        )
        linear_indices = linear_indices.clamp(
            min=0,
            max=evidence_map.shape[1] - 1,
        ).reshape(self.num_envs, -1)
        valid_values = valid.reshape(
            self.num_envs, -1
        ).to(evidence_map.dtype)

        evidence_map.scatter_add_(
            1,
            linear_indices,
            valid_values,
        )
        evidence_map.clamp_(max=1.0)

    def update_voxel_map(self, touch_sensor_positions):
        num_voxels = int(np.prod(self.voxel_grid_size))
        step_free = torch.zeros(
            self.num_envs,
            num_voxels,
            device=self.device,
        )
        step_contact = torch.zeros_like(step_free)

        object_touch = self.binary_touch > 0.5

        self.mark_voxel_evidence(
            step_free,
            touch_sensor_positions,
            torch.logical_not(object_touch),
        )
        self.mark_voxel_evidence(
            step_contact,
            touch_sensor_positions,
            object_touch,
        )

        occupancy_map = self.voxel_map[:, 0].reshape(
            self.num_envs, num_voxels
        )
        recency_map = self.voxel_map[:, 1].reshape(
            self.num_envs, num_voxels
        )
        post_contact = self.episode_had_contact.unsqueeze(1)
        recency_scale = (
            1.0
            + post_contact.to(recency_map.dtype)
            * (self.voxel_recency_decay - 1.0)
        )
        recency_map.mul_(recency_scale)
        occupancy_map.masked_fill_(
            torch.logical_and(
                post_contact,
                recency_map < self.voxel_expiration_threshold,
            ),
            -1.0,
        )

        current_contact = step_contact > 0
        current_free = torch.logical_and(
            step_free > 0,
            torch.logical_not(current_contact),
        )
        self.episode_had_valid_estimate.logical_or_(
            self.tactile_object_position_quality
            >= self.tactile_position_minimum_quality
        )
        newly_explored_free = torch.logical_and(
            current_free,
            occupancy_map < -0.5,
        )
        newly_explored_free.logical_and_(
            torch.logical_not(self.episode_had_valid_estimate).unsqueeze(1)
        )
        newly_explored_free.logical_and_(self.voxel_exploration_z_mask)
        new_voxel_count = newly_explored_free.sum(dim=1).float()
        self.voxel_exploration_reward.copy_(
            self.voxel_exploration_reward_scale
            * torch.clamp(
                new_voxel_count
                / self.voxel_exploration_new_voxel_cap,
                max=1.0,
            )
        )

        occupancy_map.masked_fill_(current_free, 0.0)
        occupancy_map.masked_fill_(current_contact, 1.0)
        recency_map.masked_fill_(
            torch.logical_or(current_free, current_contact), 1.0
        )

        return self.voxel_map.flatten(1)

    def compute_local_voxel_map(self, touch_sensor_positions):
        cell_size = (
            self.voxel_upper - self.voxel_lower
        ) / self.voxel_grid_size_tensor
        palm_index = torch.floor(
            (self.right_hand_pos - self.voxel_lower) / cell_size
        ).long()
        world_indices = palm_index.unsqueeze(1) + self.local_voxel_offsets
        inside_world = (
            (world_indices >= 0)
            & (world_indices < self.voxel_grid_size_tensor)
        ).all(dim=-1)
        clipped_indices = torch.minimum(
            torch.maximum(world_indices, torch.zeros_like(world_indices)),
            self.voxel_grid_size_tensor - 1,
        )
        world_linear = (
            (clipped_indices[..., 0] * self.voxel_grid_size[1]
             + clipped_indices[..., 1]) * self.voxel_grid_size[2]
            + clipped_indices[..., 2]
        )
        local_shape = (self.num_envs, *self.local_voxel_grid_size)
        local_map = torch.zeros(
            self.num_envs, 8, *self.local_voxel_grid_size,
            device=self.device, dtype=self.voxel_map.dtype,
        )
        world_map = self.voxel_map.flatten(2)
        for channel in range(2):
            inherited = world_map[:, channel].gather(1, world_linear)
            inherited = inherited.masked_fill(
                ~inside_world, -1.0 if channel == 0 else 0.0
            )
            local_map[:, channel] = inherited.reshape(local_shape)

        sensor_indices = torch.floor(
            (touch_sensor_positions - self.voxel_lower) / cell_size
        ).long() - palm_index.unsqueeze(1) + self.local_voxel_half_size
        inside_local = (
            (sensor_indices >= 0)
            & (sensor_indices < self.local_voxel_grid_size_tensor)
        ).all(dim=-1)
        sensor_linear = (
            (sensor_indices[..., 0] * self.local_voxel_grid_size[1]
             + sensor_indices[..., 1]) * self.local_voxel_grid_size[2]
            + sensor_indices[..., 2]
        ).clamp(0, int(np.prod(self.local_voxel_grid_size)) - 1)
        for finger in range(5):
            finger_valid = (
                inside_local
                & self.touch_sensor_finger_membership[finger].unsqueeze(0)
            )
            local_map[:, finger + 2].flatten(1).scatter_add_(
                1, sensor_linear, finger_valid.float()
            )
        local_map[:, 2:7].clamp_(max=1.0)

        estimate_indices = torch.floor(
            (self.tactile_object_position_estimate - self.voxel_lower)
            / cell_size
        ).long() - palm_index + self.local_voxel_half_size
        valid_estimate = (
            (self.tactile_object_position_quality
             >= self.tactile_position_minimum_quality)
            & (estimate_indices >= 0).all(dim=-1)
            & (estimate_indices < self.local_voxel_grid_size_tensor).all(dim=-1)
        )
        estimate_linear = (
            (estimate_indices[:, 0] * self.local_voxel_grid_size[1]
             + estimate_indices[:, 1]) * self.local_voxel_grid_size[2]
            + estimate_indices[:, 2]
        ).clamp(0, int(np.prod(self.local_voxel_grid_size)) - 1)
        local_map[:, 7].flatten(1).scatter_(
            1, estimate_linear.unsqueeze(1), valid_estimate.float().unsqueeze(1)
        )
        return local_map.flatten(1)

    def compute_tactile_observations(self):
        self.compute_binary_touch()
        if (
            self.relative_object_position_source == "tactile_estimate"
            and (self.relative_object_position_enabled
                 or self.tactile_experiment == "E3")
        ):
            self.update_tactile_object_position_estimate()
        (
            proprioception,
            proprioception_core,
            fingertip_positions,
            previous_actions,
        ) = self.compute_tactile_proprioception()

        touch_sensor_positions = self.touch_sensor_pos

        observation_branches = [
            proprioception,
            self.binary_touch,
            self.compute_relative_object_position_observation(),
        ]

        if self.tactile_experiment in {"E2", "E5"}:
            pointnet_global = (
                self.DexRepEncoder
                .get_batch_object_global_feature()
            )
            if pointnet_global.shape[1] != self.pointnet_global_dim:
                raise RuntimeError(
                    f"Expected {self.pointnet_global_dim} PointNet "
                    f"features, got {pointnet_global.shape[1]}"
                )

            observation_branches.extend(
                (
                    self.compute_oracle_object_observation(),
                    pointnet_global,
                )
            )

            if self.tactile_experiment == "E5":
                dexrep_interaction = (
                    self.DexRepEncoder.pre_observation(
                        obj_pos=self.object_pos,
                        obj_rot=self.object_rot,
                        hand_pos=self.dexrep_hand_state[:, 11, 0:3],
                        hand_rot=self.dexrep_hand_state[:, 11, 3:7],
                        joints_sate=self.dexrep_hand_pos,
                        clip_range=self.cfg["env"][
                            "clip_observations"
                        ],
                    )
                )
                if (
                    dexrep_interaction.shape[1]
                    != self.dexrep_interaction_dim
                ):
                    raise RuntimeError(
                        f"Expected {self.dexrep_interaction_dim} "
                        f"DexRep values, got "
                        f"{dexrep_interaction.shape[1]}"
                    )
                observation_branches.append(dexrep_interaction)
        elif self.tactile_experiment == "E3":
            observation_branches.append(
                self.update_voxel_map(touch_sensor_positions)
            )
            observation_branches.append(
                self.compute_local_voxel_map(touch_sensor_positions)
            )
        elif self.tactile_experiment == "E4":
            observation_branches.append(
                self.update_touch_history(
                    proprioception_core,
                    touch_sensor_positions,
                    previous_actions,
                )
            )

        observations = torch.cat(observation_branches, dim=1)
        expected_dim = sum(self.cfg["env"]["obs_dim"].values())
        if observations.shape[1] != expected_dim:
            raise RuntimeError(
                f"Expected observation dimension {expected_dim}, "
                f"got {observations.shape[1]}"
            )

        clip_observations = self.cfg["env"]["clip_observations"]
        return torch.clamp(
            observations,
            -clip_observations,
            clip_observations,
        )

    def compute_full_state(self, asymm_obs=False):
        obs_buf = torch.zeros((self.num_envs, 207), device=self.device, dtype=torch.float)
        # unscale to (-1，1)
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##

        # 0:66
        obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                               self.shadow_hand_dof_lower_limits,
                                                               self.shadow_hand_dof_upper_limits)
        obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        obs_buf[:,2 * self.num_shadow_hand_dofs:3 * self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]

        fingertip_obs_start = 3 * self.num_shadow_hand_dofs
        # 66:131: ft states
        obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)

        # 131:161: ft sensors: do not need repose
        obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]

        hand_pose_start = fingertip_obs_start + 95
        # 161:167: hand_pose
        obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.right_hand_pos
        euler_xyz = get_euler_xyz(
            self.hand_orientations[self.hand_indices, :]
        )
        obs_buf[:, hand_pose_start + 3:hand_pose_start + 4] = euler_xyz[0].unsqueeze(-1)
        obs_buf[:, hand_pose_start + 4:hand_pose_start + 5] = euler_xyz[1].unsqueeze(-1)
        obs_buf[:, hand_pose_start + 5:hand_pose_start + 6] = euler_xyz[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        # 167:191: action
        obs_buf[:, action_obs_start:action_obs_start + 24] = self.actions[:, :24]

        obj_obs_start = action_obs_start + 24  # 144
        # 191:207 object_pose, goal_pos
        obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.object_pose[:, 0:3]
        obs_buf[:, obj_obs_start + 3:obj_obs_start + 7] = self.object_pose[:, 3:7]
        obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

         # 207:236 goal
        # hand_goal_start = obj_obs_start + 16
        # obs_buf[:, hand_goal_start:hand_goal_start + 3] = self.delta_target_hand_pos
        # obs_buf[:, hand_goal_start + 3:hand_goal_start + 7] = self.delta_target_hand_rot
        # obs_buf[:, hand_goal_start + 7:hand_goal_start + 29] = self.delta_qpos

        # 236: visual feature
        # visual_feat_start = hand_goal_start + 29

        # 236: 300: visual feature
        # obs_buf[:, visual_feat_start:visual_feat_start + 64] = 0.1 * self.visual_feat_buf

        return obs_buf

    def reset_target_pose(self, env_ids, apply_reset=False):

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]

        # self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3]  # + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]

        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):
            
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)


        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_shadow_hand_dofs]

        pos = self.shadow_hand_default_dof_pos  # + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos

        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
                                               self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_shadow_hand_dofs:5 + self.num_shadow_hand_dofs * 2]

        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        all_indices = torch.unique(torch.cat([all_hand_indices, self.object_indices[env_ids], self.table_indices[env_ids], ]).to(torch.int32))  ##

        self.hand_positions[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 0:3]
        self.hand_orientations[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 3:7]

        # Keep the hand at its fixed initial root pose. Only the object yaw
        # and planar position are randomized independently.
        theta = torch_rand_float(-3.14, 3.14, (len(env_ids),1), device=self.device).squeeze(-1)
        zero_angle = torch.zeros_like(theta)
        new_object_rot = quat_from_euler_xyz(
            zero_angle, zero_angle, theta
        )

        self.hand_linvels[hand_indices.to(torch.long), :] = 0
        self.hand_angvels[hand_indices.to(torch.long), :] = 0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot  # reset object rotation
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        position_randomization = self.cfg["env"].get(
            "objectPositionRandomization", {}
        )
        if position_randomization.get("enabled", False):
            x_range = position_randomization.get(
                "xRange", [-0.15, 0.15]
            )
            y_range = position_randomization.get(
                "yRange", [-0.15, 0.15]
            )
            x_offset = torch_rand_float(
                x_range[0],
                x_range[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(-1)
            y_offset = torch_rand_float(
                y_range[0],
                y_range[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(-1)

            self.root_state_tensor[
                self.object_indices[env_ids], 0
            ] += x_offset
            self.root_state_tensor[
                self.object_indices[env_ids], 1
            ] += y_offset

            self.goal_states[env_ids] = self.goal_init_state[
                env_ids
            ].clone()
            self.goal_states[env_ids, 0] += x_offset
            self.goal_states[env_ids, 1] += y_offset
            self.root_state_tensor[
                self.goal_object_indices[env_ids]
            ] = self.goal_states[env_ids]

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              self.object_indices[env_ids],
                                              self.goal_object_indices[env_ids],
                                              self.table_indices[env_ids], ]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))

        if self.random_time:
            self.random_time = False
            self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        else:
            self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        if self.tactile_enabled:
            if self.relative_object_position_enabled:
                self.relative_object_position_visibility[env_ids] = (
                    torch.rand(
                        len(env_ids),
                        1,
                        device=self.device,
                    )
                    >= self.relative_object_position_mask_probability
                ).float()
            else:
                self.relative_object_position_visibility[env_ids] = 0
            self.binary_touch[env_ids] = 0
            self.previous_binary_touch[env_ids] = 0
            self.tactile_object_position_estimate[env_ids] = 0
            self.tactile_object_position_quality[env_ids] = 0
            self.tactile_position_previous_contact_positions[env_ids] = 0
            self.tactile_position_previous_contact_directions[env_ids] = 0
            self.tactile_position_previous_touch[env_ids] = False
            self.consecutive_valid_grasp_steps[env_ids] = 0
            self.valid_grasp_mask[env_ids] = False
            self.episode_had_contact[env_ids] = False
            self.episode_had_valid_estimate[env_ids] = False
            self.episode_had_multi_contact[env_ids] = False
            self.episode_had_lift[env_ids] = False
            self.episode_first_contact_step[env_ids] = -1.0
            self.episode_contact_steps[env_ids] = 0
            self.episode_elapsed_steps[env_ids] = 0
            self.episode_contact_losses[env_ids] = 0
            self.voxel_exploration_reward[env_ids] = 0
            self.episode_object_start_height[env_ids] = (
                self.root_state_tensor[
                    self.object_indices[env_ids], 2
                ]
            )
            self.previous_reach_distance[env_ids] = 0
            self.previous_reach_distance_valid[env_ids] = False
            self.previous_reach_keypoints[env_ids] = 0
            self.consecutive_lift_hold_steps[env_ids] = 0
            self.touch_history[env_ids] = 0
            self.voxel_map[env_ids] = 0
            self.voxel_map[env_ids, 0] = -1.0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)

        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 6:],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:,self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])


            self.apply_forces[:, self.hand_body_idx_dict["palm"], :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
            self.apply_torque[:, self.hand_body_idx_dict["palm"], :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces),
                                                    gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions, self.id)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.object_back_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.goal_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_pos[i], self.right_hand_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_th_pos[i], self.right_hand_th_rot[i])

                # self.add_debug_lines(self.envs[i], self.left_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_th_pos[i], self.right_hand_th_rot[i])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
        object_init_z, object_start_height, touch_count,
        lift_target_height: float, lift_hold_reward_scale: float,
        lift_hold_steps, lift_hold_steps_required: float,
        id: int, object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, current_successes, consecutive_successes,
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,
        right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int,
        av_factor: float, goal_cond: bool,
        tactile_progress_reward: bool
):
    action_penalty = action_penalty_scale * torch.sum(
        actions * actions,
        dim=-1,
    )
    if tactile_progress_reward:
        # Tactile experiments add signed distance progress outside this
        # function. Do not retain the repeatable legacy proximity reward.
        reward = action_penalty
    else:
        # Preserve the original reward for non-tactile experiments.
        goal_dist = torch.norm(
            target_pos - object_pos,
            p=2,
            dim=-1,
        )
        right_hand_dist = torch.norm(
            object_handle_pos - right_hand_pos,
            p=2,
            dim=-1,
        )
        right_hand_dist = torch.clamp(right_hand_dist, max=0.5)
        right_hand_finger_dist = (
            torch.norm(
                object_handle_pos - right_hand_ff_pos,
                p=2,
                dim=-1,
            )
            + torch.norm(
                object_handle_pos - right_hand_mf_pos,
                p=2,
                dim=-1,
            )
            + torch.norm(
                object_handle_pos - right_hand_rf_pos,
                p=2,
                dim=-1,
            )
            + torch.norm(
                object_handle_pos - right_hand_lf_pos,
                p=2,
                dim=-1,
            )
            + torch.norm(
                object_handle_pos - right_hand_th_pos,
                p=2,
                dim=-1,
            )
        )
        right_hand_finger_dist = torch.clamp(
            right_hand_finger_dist,
            max=3.0,
        )
        hand_near_object = torch.logical_and(
            right_hand_finger_dist <= 0.6,
            right_hand_dist <= 0.12,
        )
        goal_hand_reward = torch.where(
            hand_near_object,
            0.9 - 2.0 * goal_dist,
            torch.zeros_like(goal_dist),
        )
        reward = (
            -0.5 * right_hand_finger_dist
            - right_hand_dist
            + goal_hand_reward
            + action_penalty
        )

    lift_amount = object_pos[:, 2] - object_start_height
    planar_goal_dist = torch.norm(
        target_pos[:, 0:2] - object_pos[:, 0:2],
        p=2,
        dim=-1,
    )
    if tactile_progress_reward:
        goal_reached = (
            (lift_amount >= lift_target_height)
            & (touch_count >= 2.0)
        )
    else:
        goal_reached = (
            (lift_amount >= lift_target_height)
            & (touch_count >= 2.0)
            & (planar_goal_dist <= success_tolerance)
        )

    new_success = goal_reached & (successes < 0.5)
    reward = (
        reward
        + reach_goal_bonus * new_success.float()
    )
    if tactile_progress_reward:
        reward = reward + (
            lift_hold_reward_scale * goal_reached.float()
        )

    successes = torch.where(
        goal_reached,
        torch.ones_like(successes),
        successes,
    )

    resets = reset_buf

    timeout = progress_buf >= max_episode_length
    if tactile_progress_reward:
        stable_lift = (
            lift_hold_steps >= lift_hold_steps_required
        )
        resets = torch.where(
            timeout | stable_lift,
            torch.ones_like(resets),
            resets,
        )
    else:
        resets = torch.where(
            timeout | goal_reached,
            torch.ones_like(resets),
            resets,
        )

    goal_resets = resets
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    current_successes = torch.where(resets, successes, current_successes)
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
