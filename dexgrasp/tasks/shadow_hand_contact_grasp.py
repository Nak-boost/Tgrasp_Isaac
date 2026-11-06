from pathlib import Path

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from tasks.shadow_hand_grasp_dexrep import ShadowHandGraspDexRep
from utils.torch_jit_utils import quat_conjugate, quat_from_euler_xyz, quat_mul, unscale


class ShadowHandContactGrasp(ShadowHandGraspDexRep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sweep_cfg = self.cfg['env']['contact_grasp']
        if self.tactile_experiment != 'E2' or self.voxel_enabled:
            raise ValueError('Contact grasp requires E2 history without voxel observations')
        if self.actor_uses_oracle_position or self.relative_object_position_source != 'tactile_estimate':
            raise ValueError('The contact-grasp actor must use tactile position estimates in training and inference')
        self.grasp_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.policy_training_mask = torch.zeros(self.num_envs, device=self.device)
        self.sweep_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.handoff_step = torch.full_like(self.sweep_steps, -1)
        self.sweep_origin = self.hand_positions[self.hand_indices].clone()
        self.sweep_rotation = self.hand_orientations[self.hand_indices].clone()
        self.sweep_joints = self.shadow_hand_default_dof_pos.clone()
        names = self.gym.get_actor_dof_names(self.envs[0], self.shadow_hands[0])
        for index, name in enumerate(names):
            self.sweep_joints[index] = self.sweep_cfg['joint_positions'].get(name.split(':')[-1], 0.)
        self.sweep_joints.clamp_(self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.shadow_hand_default_dof_pos.copy_(self.sweep_joints)
        self.mesh_bounds = []
        for name in self.object_code_list:
            vertices = []
            for path in (Path(self.obj_asset_root) / ('grab-' + name) / 'coacd').glob('coacd_convex_piece_*.obj'):
                vertices.extend([float(value) for value in line.split()[1:4]]
                                for line in path.read_text().splitlines() if line.startswith('v '))
            if not vertices:
                raise ValueError('Missing collision geometry: ' + name)
            vertices = np.asarray(vertices)
            self.mesh_bounds.append([vertices.min(axis=0), vertices.max(axis=0)])
        self.mesh_bounds = torch.tensor(np.asarray(self.mesh_bounds), device=self.device, dtype=torch.float)

    def reset(self, env_ids, goal_env_ids):
        super().reset(env_ids, goal_env_ids)
        hand_ids = self.hand_indices[env_ids]
        object_ids = self.object_indices[env_ids]
        count = len(env_ids)
        roll = torch.full((count,), np.deg2rad(self.sweep_cfg['roll_degrees']), device=self.device)
        rotation = quat_from_euler_xyz(roll, torch.zeros_like(roll), torch.full_like(roll, -np.pi / 2))
        self.root_state_tensor[hand_ids, :3] = self.root_state_tensor.new_tensor([
            self.sweep_cfg['start_x'], 0., self.table_top_z + self.sweep_cfg['height_above_table']])
        self.root_state_tensor[hand_ids, 3:7] = rotation
        self.root_state_tensor[hand_ids, 7:] = 0
        self.sweep_origin[env_ids] = self.root_state_tensor[hand_ids, :3]
        self.sweep_rotation[env_ids] = rotation
        bounds = self.mesh_bounds[env_ids % len(self.object_code_list)]
        object_rotation = self.root_state_tensor[object_ids, 3:7]
        cosine = 1 - 2 * object_rotation[:, 2].square()
        sine = 2 * object_rotation[:, 2] * object_rotation[:, 3]
        min_x = torch.minimum(cosine * bounds[:, 0, 0], cosine * bounds[:, 1, 0])
        min_x += torch.minimum(-sine * bounds[:, 0, 1], -sine * bounds[:, 1, 1])
        gap_low, gap_high = self.sweep_cfg['forward_gap_range']
        gap = gap_low + (gap_high - gap_low) * torch.rand(count, device=self.device)
        self.root_state_tensor[object_ids, 0] = self.sweep_cfg['start_x'] + .04 + gap - min_x
        self.root_state_tensor[object_ids, 1] = (torch.rand(count, device=self.device) * 2 - 1) * self.sweep_cfg['lateral_range']
        self.root_state_tensor[object_ids, 2] = self.table_top_z - bounds[:, 0, 2] + .002
        self.root_state_tensor[object_ids, 7:] = 0
        self.episode_object_start_height[env_ids] = self.root_state_tensor[object_ids, 2]
        self.root_state_tensor[self.goal_object_indices[env_ids], :3] = self.root_state_tensor[object_ids, :3] + self.goal_displacement_tensor
        self.goal_states[env_ids] = self.root_state_tensor[self.goal_object_indices[env_ids]]
        indices = torch.cat((hand_ids, object_ids, self.goal_object_indices[env_ids])).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                                                   gymtorch.unwrap_tensor(indices), len(indices))
        self.grasp_phase[env_ids] = False
        self.policy_training_mask[env_ids] = 0
        self.sweep_steps[env_ids] = 0
        self.handoff_step[env_ids] = -1

    def update_touch_history(self, proprioception_core, touch_sensor_positions, previous_actions):
        reference = self.sweep_origin.clone()
        reference[:, 2] = self.table_top_z
        local_core = proprioception_core.clone()
        position_start = 2 * self.tactile_hand_dof_dim
        local_core[:, position_start:position_start + 3] -= reference
        return super().update_touch_history(
            local_core, touch_sensor_positions - reference.unsqueeze(1), previous_actions,
        )

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self.policy_training_mask.copy_(self.grasp_phase.float())
        sweeping = ~self.grasp_phase
        elapsed = (self.sweep_steps - self.sweep_cfg['settle_steps']).clamp(min=0).float() * self.dt
        target = self.sweep_origin.clone()
        target[:, 0] += (elapsed * self.sweep_cfg['speed']).clamp(max=self.sweep_cfg['distance'])
        velocity = torch.zeros_like(target)
        velocity[:, 0] = ((self.sweep_steps >= self.sweep_cfg['settle_steps'])
                          & (elapsed * self.sweep_cfg['speed'] < self.sweep_cfg['distance'])).float() * self.sweep_cfg['speed']
        hand = self.root_state_tensor[self.hand_indices]
        force = self.sweep_cfg['position_kp'] * (target - hand[:, :3]) + self.sweep_cfg['position_kd'] * (velocity - hand[:, 7:10])
        error = quat_mul(self.sweep_rotation, quat_conjugate(hand[:, 3:7]))
        sign = torch.where(error[:, 3:] < 0, -torch.ones_like(error[:, 3:]), torch.ones_like(error[:, 3:]))
        torque = self.sweep_cfg['rotation_kp'] * 2 * error[:, :3] * sign - self.sweep_cfg['rotation_kd'] * hand[:, 10:13]
        force_limit = self.sweep_cfg['force_limit']
        force = force.clamp(-force_limit, force_limit)
        torque = torque.clamp(-self.sweep_cfg['torque_limit'], self.sweep_cfg['torque_limit'])
        palm = self.hand_body_idx_dict['palm']
        self.apply_forces[sweeping, palm] = force[sweeping]
        self.apply_torque[sweeping, palm] = torque[sweeping]
        self.actions[sweeping, :3] = force[sweeping] / (self.dt * self.transition_scale * 100000)
        self.actions[sweeping, 3:6] = torque[sweeping] / (self.dt * self.orientation_scale * 1000)
        joints = self.actuated_dof_indices
        joint_commands = unscale(self.sweep_joints[joints], self.shadow_hand_dof_lower_limits[joints], self.shadow_hand_dof_upper_limits[joints])
        self.actions[sweeping, 6:] = joint_commands
        self.cur_targets[sweeping, :self.num_shadow_hand_dofs] = self.sweep_joints
        self.prev_targets[sweeping, :self.num_shadow_hand_dofs] = self.sweep_joints
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces),
                                               gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)
        indices = self.hand_indices.to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                       gymtorch.unwrap_tensor(indices), len(indices))

    def compute_reward(self, actions, id=-1):
        super().compute_reward(actions, id)
        self.sweep_steps.add_((~self.grasp_phase).long())
        contact = (self.binary_touch > .5).any(dim=-1)
        handoff = ~self.grasp_phase & contact
        self.handoff_step[handoff] = self.progress_buf[handoff]
        self.episode_object_start_height[handoff] = self.object_pos[handoff, 2]
        self.grasp_phase.logical_or_(handoff)
        missed = ~self.grasp_phase & (self.sweep_steps >= self.sweep_cfg['max_steps'])
        self.reset_buf.masked_fill_(missed, 1)
        self.rew_buf.mul_(self.policy_training_mask)
        self.extras['policy_training_mask'] = self.policy_training_mask.clone()
        self.extras['grasp_handoff'] = handoff.float()
        self.extras['sweep_missed'] = missed.float()
        self.extras['handoff_step'] = self.handoff_step.float()
        self.extras['grasp_phase'] = self.grasp_phase.float()
