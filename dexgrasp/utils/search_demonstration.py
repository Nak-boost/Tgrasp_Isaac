import math

import numpy as np
import torch


def serpentine_waypoints(start, sensor_offset, bounds, sensor_height, lane_spacing):
    xmin, xmax, ymin, ymax = bounds
    if lane_spacing <= 0 or xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid search bounds or lane spacing")
    lane_count = int(math.ceil((xmax - xmin) / lane_spacing)) + 1
    points = [np.asarray(start, dtype=np.float64)]
    lowered = points[0].copy()
    lowered[2] = sensor_height - sensor_offset[2]
    points.append(lowered)
    for lane_index, coordinate in enumerate(np.linspace(xmin, xmax, lane_count)):
        endpoints = (ymin, ymax) if lane_index % 2 == 0 else (ymax, ymin)
        for endpoint in endpoints:
            points.append(np.array([coordinate, endpoint, sensor_height]) - sensor_offset)
    return np.asarray(points)


def sample_path(waypoints, speed, dt, return_phases=False):
    if speed <= 0 or dt <= 0:
        raise ValueError("Speed and timestep must be positive")
    positions, velocities, phases = [], [], []
    for segment, (start, end) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        displacement = end - start
        duration = max(1.5 * np.linalg.norm(displacement) / speed, 0.3)
        count = max(2, int(math.ceil(duration / dt)))
        duration = count * dt
        phase = np.arange(1, count + 1) / count
        blend = 3 * phase ** 2 - 2 * phase ** 3
        rate = (6 * phase - 6 * phase ** 2) / duration
        positions.extend(start + blend[:, None] * displacement)
        velocities.extend(rate[:, None] * displacement)
        phase_id = 1 if segment == 0 else 2 if segment == 1 else 3 if abs(displacement[1]) > abs(displacement[0]) else 4
        phases.extend([phase_id] * count)
    result = (np.asarray(positions, dtype=np.float32), np.asarray(velocities, dtype=np.float32))
    return (*result, np.asarray(phases, dtype=np.int64)) if return_phases else result


def orientation_error(target, current):
    target_vector, target_scalar = target[:, :3], target[:, 3:]
    current_vector, current_scalar = current[:, :3], current[:, 3:]
    relative_vector = (
        current_scalar * target_vector - target_scalar * current_vector
        - torch.cross(target_vector, current_vector, dim=-1)
    )
    relative_scalar = (target * current).sum(dim=-1, keepdim=True)
    return 2 * relative_vector * torch.where(relative_scalar >= 0, 1.0, -1.0)


class BasePathController:
    def __init__(self, task, kp=4000.0, kd=80.0, drag=2650.0, rotation_kp=12.0, rotation_kd=1.5):
        self.task = task
        self.kp = kp
        self.kd = kd
        self.drag = drag
        self.rotation_kp = rotation_kp
        self.rotation_kd = rotation_kd
        self.force_scale = task.dt * task.transition_scale * 100000
        self.torque_scale = task.dt * task.orientation_scale * 1000
        if min(self.force_scale, self.torque_scale) <= 0:
            raise ValueError("Base action scales must be positive")

    def actions(self, target_position, target_velocity, target_orientation, finger_actions):
        state = self.task.rigid_body_states[:, self.task.hand_body_idx_dict["palm"]]
        force = (
            self.kp * (target_position - state[:, :3])
            + self.kd * (target_velocity - state[:, 7:10])
            + self.drag * target_velocity
        )
        torque = (
            self.rotation_kp * orientation_error(target_orientation, state[:, 3:7])
            - self.rotation_kd * state[:, 10:13]
        )
        base_actions = torch.cat((force / self.force_scale, torque / self.torque_scale), dim=-1)
        return torch.cat((base_actions.clamp(-1, 1), finger_actions), dim=-1)
