# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from utils.torch_jit_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi
from tasks.shadow_hand_grasp_dexrep import ShadowHandGraspDexRep
class ShadowHandGraspDexRepDexgrasp(ShadowHandGraspDexRep):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        if cfg['env'].get('fric_type') == 'fixed':
            self.obj_fric = cfg['env'].get("obj_fric", 1.0)
            print("obj_fric: ", self.obj_fric)
            self.obj_fric_type = 'fixed'
        elif cfg['env'].get('fric_type') == 'random':
            self.obj_fric = cfg['env'].get("obj_fric", 1.0)
            print('obj_fric is random')
            self.obj_fric_type = 'random'
        else:
            self.obj_fric = 1.0
            print('set obj_fric to default value: ', 1.0)
            self.obj_fric_type = 'fixed'
        super(ShadowHandGraspDexRepDexgrasp, self).__init__(cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=agent_index, is_multi_agent=is_multi_agent)

    def _create_envs(self, num_envs, spacing, num_per_row):
   
        object_code_list = self.cfg['env']['object_code_dict']
        num_objs = self.cfg['env']['num_objs']
        if isinstance(object_code_list, dict):
            keys = list(object_code_list.keys())
            keys_ = keys[::len(object_code_list)//num_objs]
            self.object_code_list = dict()
            for k in keys_[:num_objs]:
                self.object_code_list[k] = object_code_list[k]
        else:
            self.object_code_list = object_code_list[:num_objs]
        self.repose_z = self.cfg['env']['repose_z']

        self.grasp_data = {}
        assets_path = '../assets'

        print(f'Num Objs: {len(self.object_code_list)}')
        print(f'Num Envs: {self.num_envs}')

        self.goal_cond = self.cfg["env"]["goal_cond"]
        self.random_prior = self.cfg['env']['random_prior']
        self.random_time = self.cfg["env"]["random_time"]
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
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
        pose_dx, pose_dy, pose_dz = -1.0, 0.0, -0.0

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
            object_shape_props[0].friction = self.obj_fric
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


    def _load_object_asset(self, assets_path):
        object_asset_dict = {}
        goal_asset_dict = {}
        self.num_object_bodies_list = []
        self.num_object_shapes_list = []
        # mesh_path = osp.join(assets_path, 'meshdatav3_scaled')
        self.asset_root = self.cfg["env"]["asset"]["assetRoot"]
        self.obj_asset_root = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj"]
        self.raw_obj_asset_root = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj_raw"]
        for object_id, (object_code, obj_scale) in enumerate(self.object_code_list.items() if isinstance(self.object_code_list, dict) else zip(self.object_code_list, self.object_code_list)):
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
            scale2str = {
                0.06: '006',
                0.08: '008',
                0.10: '010',
                0.12: '012',
                0.15: '015',
            }
            if isinstance(self.object_code_list, list):
                obj_scale = object_code.split('-')[-1]
                object_code = object_code[:-(len(obj_scale)+1)]
                object_asset_file = f"coacd_{scale2str[float(obj_scale)]}.urdf"
                dexrep_load = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj_raw"] + object_code + "/coacd" + f'/decomposed_{scale2str[float(obj_scale)]}.obj'
            else:
                object_asset_file = f"coacd_{scale2str[obj_scale[0]]}.urdf"
                dexrep_load = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj_raw"] + object_code + "/coacd" + f'/decomposed_{scale2str[obj_scale[0]]}.obj'

            object_asset = self.gym.load_asset(self.sim, self.obj_asset_root + object_code + "/coacd", object_asset_file, object_asset_options)
            if object_asset is None:
                print(object_code)
            assert object_asset is not None

            object_asset_options.disable_gravity = True
            goal_asset = self.gym.create_sphere(self.sim, 0.005, object_asset_options)

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

    def reset(self, env_ids, goal_env_ids):

        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5 + 1), device=self.device)
        if self.obj_fric_type == "random":
            self.reset_obj_fric(env_ids, rand_floats)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_shadow_hand_dofs]

        pos = self.shadow_hand_default_dof_pos  # + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos

        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
                                               self.reset_dof_vel_noise * rand_floats[:,
                                                                          5 + self.num_shadow_hand_dofs:5 + self.num_shadow_hand_dofs * 2]

        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        all_indices = torch.unique(
            torch.cat([all_hand_indices, self.object_indices[env_ids], self.table_indices[env_ids], ]).to(
                torch.int32))  ##

        self.hand_positions[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 0:3]
        self.hand_orientations[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 3:7]

        theta = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device)[:, 0]

        # reset obejct with all data:
        new_object_rot = quat_from_euler_xyz(self.object_init_euler_xy[env_ids, 0],
                                             self.object_init_euler_xy[env_ids, 1], theta)
        prior_rot_z = get_euler_xyz(quat_mul(new_object_rot, self.target_hand_rot[env_ids]))[2]

        # coordinate transform according to theta(object)/ prior_rot_z(hand)
        self.z_theta[env_ids] = prior_rot_z
        prior_rot_quat = quat_from_euler_xyz(torch.tensor(1.57, device=self.device).repeat(len(env_ids), 1)[:, 0],
                                             torch.zeros_like(theta), prior_rot_z)

        self.hand_orientations[hand_indices.to(torch.long), :] = prior_rot_quat
        self.hand_linvels[hand_indices.to(torch.long), :] = 0
        self.hand_angvels[hand_indices.to(torch.long), :] = 0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot  # reset object rotation
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13])

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              self.object_indices[env_ids],
                                              self.goal_object_indices[env_ids],
                                              self.table_indices[env_ids], ]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))

        if self.random_time:
            self.random_time = False
            self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        else:
            self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def reset_obj_fric(self, env_ids, rand_floats):
        rand_fric = (rand_floats[:, -1] + 1) / 2
        for env_id in range(len(env_ids)):
            object_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.objects[env_id])
            object_shape_props[0].friction = rand_fric[env_id]
            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], self.objects[env_id], object_shape_props)


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