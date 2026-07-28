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

        self.use_dexrep = False
        self.use_pnG = False
        self.use_geodex = False

        if self.tactile_enabled:
            valid_experiments = {"E1", "E2", "E3", "E4"}
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
            self.current_touch_dim = 5
            self.oracle_object_dim = 13
            self.tactile_core_dim = (
                2 * self.tactile_hand_dof_dim + 6
            )
            self.tactile_prop_dim = (
                self.tactile_core_dim
                + 3 * self.current_touch_dim
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
            }

            if self.tactile_experiment == "E2":
                obs_dim["oracle_object"] = self.oracle_object_dim
            elif self.tactile_experiment == "E3":
                voxel_cfg = self.tactile_cfg["voxel"]
                voxel_size = int(np.prod(voxel_cfg["grid_size"]))
                obs_dim["voxel_map"] = (
                    voxel_cfg.get("channels", 3) * voxel_size
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
        if self.tactile_enabled:
            self.use_dexrep = False
        elif self.obs_type in _DexRepEncoder_Map.keys():
            assert "dexrep" in cfg.keys()
            self.use_dexrep = True
            self.DexRepEncoder = _DexRepEncoder_Map[self.obs_type](cfg, device_type+f":{device_id}")

        self.num_hand_obs = 66 + 95 + 24 + 6  # 191 =  22*3 + (65+30) + 24
        self.up_axis = 'z'
        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal",
                           "robot0:thdistal"]
        self.hand_center = ["robot0:palm"]
        self.num_fingertips = len(self.fingertips) 
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

        # if self.obs_type == "full_state" or self.asymmetric_obs:
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

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
                self.num_fingertips,
                device=self.device,
                dtype=torch.float,
            )
            self.previous_binary_touch = torch.zeros_like(
                self.binary_touch
            )

            touch_reward_cfg = self.tactile_cfg.get("reward", {})
            self.touch_contact_reward_scale = touch_reward_cfg.get(
                "contact", 0.05
            )
            self.touch_multi_reward_scale = touch_reward_cfg.get(
                "multi_contact", 0.02
            )
            self.touch_hold_reward_scale = touch_reward_cfg.get(
                "contact_hold", 0.01
            )
            self.touch_loss_penalty_scale = touch_reward_cfg.get(
                "contact_loss", 0.05
            )
            self.touch_lift_height = touch_reward_cfg.get(
                "lift_height", 0.80
            )

            self.episode_had_contact = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.bool,
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
            self.voxel_channels = voxel_cfg.get("channels", 3)
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
            if self.voxel_channels != 3:
                raise ValueError(
                    "The tactile voxel map requires three channels: "
                    "potential, contact and age"
                )
            self.voxel_map[:, 0] = 1.0
            self.voxel_map[:, 2] = -1.0
            self.previous_fingertip_positions = torch.zeros(
                self.num_envs,
                self.num_fingertips,
                3,
                device=self.device,
            )
            self.previous_fingertip_positions_valid = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.bool,
            )
            self.free_voxel_offsets = self.create_voxel_offsets(
                int(voxel_cfg.get("free_radius_voxels", 0))
            )
            self.contact_voxel_offsets = (
                self.create_voxel_offsets(
                    int(voxel_cfg.get("contact_radius_voxels", 1))
                )
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
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
        
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

        # create fingertip force sensors, if needed
        # if self.obs_type == "full_state" or self.asymmetric_obs:
        sensor_pose = gymapi.Transform()
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)

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


            # randomize colors and textures for rigid body
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, shadow_hand_actor)
            hand_color = [147/255, 215/255, 160/255]
            hand_rigid_body_index = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
            for n in self.agent_index[0]:
                for m in n:
                    for o in hand_rigid_body_index[m]:
                        self.gym.set_rigid_body_color(env_ptr, shadow_hand_actor, o, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(*hand_color))
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
        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
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
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.object_init_z,
            self.id, self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
            self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
            self.goal_pos, self.goal_rot,
            self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
            self.right_hand_lf_pos, self.right_hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor,self.goal_cond
        )

        if self.tactile_enabled:
            self.rew_buf.add_(self.compute_touch_reward())

        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes
        if self.tactile_enabled:
            self.extras["contact_found"] = (
                self.episode_had_contact.float()
            )
            self.extras["first_contact_step"] = (
                self.episode_first_contact_step
            )
            self.extras["multi_contact"] = (
                self.episode_had_multi_contact.float()
            )
            self.extras["lifted"] = self.episode_had_lift.float()
            self.extras["contact_step_ratio"] = (
                self.episode_contact_steps
                / torch.clamp(self.episode_elapsed_steps, min=1.0)
            )
            self.extras["contact_losses"] = (
                self.episode_contact_losses
            )

    def compute_touch_reward(self):
        current_touch = self.binary_touch > 0.5
        previous_touch = self.previous_binary_touch > 0.5

        touch_count = current_touch.sum(dim=1).float()
        has_contact = touch_count > 0
        additional_contacts = torch.clamp(
            touch_count - 1.0,
            min=0.0,
        )
        held_contacts = torch.logical_and(
            current_touch,
            previous_touch,
        ).sum(dim=1).float()
        lost_contacts = torch.logical_and(
            previous_touch,
            torch.logical_not(current_touch),
        ).sum(dim=1).float()

        current_episode_step = self.episode_elapsed_steps + 1.0
        first_contact = torch.logical_and(
            has_contact,
            torch.logical_not(self.episode_had_contact),
        )
        self.episode_first_contact_step = torch.where(
            first_contact,
            current_episode_step,
            self.episode_first_contact_step,
        )
        self.episode_had_contact.logical_or_(has_contact)
        self.episode_had_multi_contact.logical_or_(touch_count >= 2)
        self.episode_had_lift.logical_or_(
            self.object_pos[:, 2] >= self.touch_lift_height
        )
        self.episode_contact_steps.add_(has_contact.float())
        self.episode_elapsed_steps.copy_(current_episode_step)
        self.episode_contact_losses.add_(lost_contacts)

        touch_reward = (
            self.touch_contact_reward_scale
            * has_contact.float()
            + self.touch_multi_reward_scale
            * additional_contacts
            + self.touch_hold_reward_scale
            * held_contacts
            - self.touch_loss_penalty_scale
            * lost_contacts
        )

        self.previous_binary_touch.copy_(self.binary_touch)
        return touch_reward

    def compute_observations(self):
        # TODO:using dexrep
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # if self.obs_type == "full_state" or self.asymmetric_obs:
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
        force_vectors = self.vec_sensor_tensor.view(
            self.num_envs, self.num_fingertips, 6
        )[:, :, 0:3]
        force_norm = torch.norm(force_vectors, p=2, dim=-1)

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
        remain_touching = force_norm >= touch_off_threshold
        begin_touching = force_norm >= touch_on_threshold

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

    def compute_oracle_object_observation(self):
        return torch.cat(
            (
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.vel_obs_scale * self.object_angvel,
            ),
            dim=1,
        )

    def update_touch_history(
        self,
        proprioception_core,
        fingertip_positions,
        previous_actions,
    ):
        history_frame = torch.cat(
            (
                fingertip_positions.reshape(self.num_envs, -1),
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

    def create_voxel_offsets(self, radius_voxels):
        if radius_voxels < 0:
            raise ValueError(
                "Voxel evidence radius must be non-negative"
            )

        offset_axis = torch.arange(
            -radius_voxels,
            radius_voxels + 1,
            device=self.device,
            dtype=torch.long,
        )
        diameter = offset_axis.numel()
        offset_x = offset_axis.view(
            diameter, 1, 1
        ).expand(diameter, diameter, diameter)
        offset_y = offset_axis.view(
            1, diameter, 1
        ).expand(diameter, diameter, diameter)
        offset_z = offset_axis.view(
            1, 1, diameter
        ).expand(diameter, diameter, diameter)

        return torch.stack(
            (offset_x, offset_y, offset_z),
            dim=-1,
        ).reshape(-1, 3)

    def mark_voxel_evidence(
        self,
        evidence_map,
        positions,
        position_valid,
        voxel_offsets,
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

        offset_indices = (
            voxel_indices.unsqueeze(-2)
            + voxel_offsets.view(1, 1, -1, 3)
        )
        inside_map = torch.logical_and(
            offset_indices >= 0,
            offset_indices
            < self.voxel_grid_size_tensor.view(1, 1, 1, 3),
        ).all(dim=-1)
        valid = torch.logical_and(
            position_valid.unsqueeze(-1),
            inside_map,
        )

        linear_indices = (
            offset_indices[..., 0]
            * self.voxel_grid_size[1]
            * self.voxel_grid_size[2]
            + offset_indices[..., 1]
            * self.voxel_grid_size[2]
            + offset_indices[..., 2]
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

    def update_voxel_map(self, fingertip_positions):
        voxel_cfg = self.tactile_cfg["voxel"]
        num_voxels = int(np.prod(self.voxel_grid_size))
        step_free = torch.zeros(
            self.num_envs,
            num_voxels,
            device=self.device,
        )
        step_contact = torch.zeros_like(step_free)

        previous_positions = torch.where(
            self.previous_fingertip_positions_valid.view(
                self.num_envs, 1, 1
            ),
            self.previous_fingertip_positions,
            fingertip_positions,
        )
        sweep_samples = max(
            int(voxel_cfg.get("sweep_samples", 8)),
            2,
        )
        sweep_fraction = torch.linspace(
            0.0,
            1.0,
            sweep_samples,
            device=self.device,
        ).view(1, 1, sweep_samples, 1)
        swept_positions = (
            previous_positions.unsqueeze(2)
            + (
                fingertip_positions - previous_positions
            ).unsqueeze(2)
            * sweep_fraction
        )

        minimum_touch_z = voxel_cfg.get(
            "minimum_object_touch_z", 0.62
        )
        object_touch = torch.logical_and(
            self.binary_touch > 0.5,
            fingertip_positions[:, :, 2] >= minimum_touch_z,
        )

        contact_exclusion_samples = max(
            int(voxel_cfg.get("contact_exclusion_samples", 1)),
            1,
        )
        contact_exclusion_samples = min(
            contact_exclusion_samples,
            sweep_samples,
        )
        sample_before_contact = (
            torch.arange(
                sweep_samples,
                device=self.device,
            )
            < sweep_samples - contact_exclusion_samples
        ).view(1, 1, sweep_samples)
        free_valid = torch.logical_or(
            torch.logical_not(object_touch).unsqueeze(-1),
            sample_before_contact,
        )

        self.mark_voxel_evidence(
            step_free,
            swept_positions,
            free_valid,
            self.free_voxel_offsets,
        )
        self.mark_voxel_evidence(
            step_contact,
            fingertip_positions,
            object_touch,
            self.contact_voxel_offsets,
        )

        potential_map = self.voxel_map[:, 0].reshape(
            self.num_envs, num_voxels
        )
        contact_map = self.voxel_map[:, 1].reshape(
            self.num_envs, num_voxels
        )
        age_map = self.voxel_map[:, 2].reshape(
            self.num_envs, num_voxels
        )

        contact_decay = voxel_cfg.get("contact_decay", 0.95)
        potential_recovery = voxel_cfg.get(
            "potential_recovery", 0.005
        )
        max_age = max(voxel_cfg.get("max_age", 64), 1)

        contact_map.mul_(contact_decay)
        potential_map.add_(
            potential_recovery * (1.0 - potential_map)
        )

        previously_observed = age_map >= 0.0
        incremented_age = torch.clamp(
            age_map + 1.0 / float(max_age),
            max=1.0,
        )
        age_map.copy_(
            torch.where(
                previously_observed,
                incremented_age,
                age_map,
            )
        )

        current_contact = step_contact > 0
        current_free = torch.logical_and(
            step_free > 0,
            torch.logical_not(current_contact),
        )

        potential_map.masked_fill_(current_free, 0.0)
        contact_map.masked_fill_(current_free, 0.0)
        potential_map.masked_fill_(current_contact, 1.0)
        contact_map.copy_(
            torch.maximum(contact_map, step_contact)
        )

        current_evidence = torch.logical_or(
            current_free,
            current_contact,
        )
        age_map.masked_fill_(current_evidence, 0.0)

        self.previous_fingertip_positions.copy_(
            fingertip_positions
        )
        self.previous_fingertip_positions_valid[:] = True

        return self.voxel_map.flatten(1)

    def compute_tactile_observations(self):
        self.compute_binary_touch()
        (
            proprioception,
            proprioception_core,
            fingertip_positions,
            previous_actions,
        ) = self.compute_tactile_proprioception()

        observation_branches = [
            proprioception,
            self.binary_touch,
        ]

        if self.tactile_experiment == "E2":
            observation_branches.append(
                self.compute_oracle_object_observation()
            )
        elif self.tactile_experiment == "E3":
            observation_branches.append(
                self.update_voxel_map(fingertip_positions)
            )
        elif self.tactile_experiment == "E4":
            observation_branches.append(
                self.update_touch_history(
                    proprioception_core,
                    fingertip_positions,
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
            self.binary_touch[env_ids] = 0
            self.previous_binary_touch[env_ids] = 0
            self.episode_had_contact[env_ids] = False
            self.episode_had_multi_contact[env_ids] = False
            self.episode_had_lift[env_ids] = False
            self.episode_first_contact_step[env_ids] = -1.0
            self.episode_contact_steps[env_ids] = 0
            self.episode_elapsed_steps[env_ids] = 0
            self.episode_contact_losses[env_ids] = 0
            self.touch_history[env_ids] = 0
            self.voxel_map[env_ids] = 0
            self.voxel_map[env_ids, 0] = 1.0
            self.voxel_map[env_ids, 2] = -1.0
            self.previous_fingertip_positions[env_ids] = 0
            self.previous_fingertip_positions_valid[env_ids] = False

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


            self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
            self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000

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
        object_init_z,
        id: int, object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, current_successes, consecutive_successes,
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,
        right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.norm(object_handle_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.where(right_hand_dist >= 0.5, 0.5 + 0 * right_hand_dist, right_hand_dist)

    right_hand_finger_dist = (torch.norm(object_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
        object_handle_pos - right_hand_mf_pos, p=2, dim=-1)+ torch.norm(object_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(
                object_handle_pos - right_hand_lf_pos, p=2, dim=-1) + torch.norm(object_handle_pos - right_hand_th_pos, p=2, dim=-1))
    right_hand_finger_dist = torch.where(right_hand_finger_dist >= 3.0, 3.0 + 0 * right_hand_finger_dist,right_hand_finger_dist)
    lowest = object_pos[:, 2]


    flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()
    goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
    goal_hand_rew = torch.where(flag == 2, 1 * (0.9 - 2 * goal_dist), goal_hand_rew)

    hand_up = torch.zeros_like(right_hand_finger_dist)
    hand_up = torch.where(lowest >= 0.630, torch.where(flag == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
    hand_up = torch.where(lowest >= 0.80, torch.where(flag == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

    flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()
    bonus = torch.zeros_like(goal_dist)
    bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus)

    reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus
    
    
    resets = reset_buf

    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = resets
    successes = torch.where(goal_dist <= 0.05, torch.ones_like(successes), successes)
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