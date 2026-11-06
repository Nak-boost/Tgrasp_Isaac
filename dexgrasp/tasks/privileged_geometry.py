import numpy as np
import torch
import trimesh

from utils.torch_jit_utils import quat_apply, quat_conjugate


def sample_object_surface(path, num_points=256, seed=0):
    mesh = trimesh.load(path, force="mesh", process=False)
    areas = np.asarray(mesh.area_faces)
    if not np.isfinite(areas).all() or areas.sum() <= 0 or num_points <= 0:
        raise ValueError("Invalid object surface or point count: " + str(path))
    random = np.random.RandomState(seed)
    faces = random.choice(len(areas), size=num_points, p=areas / areas.sum())
    coordinates = random.uniform(size=(num_points, 2))
    root = np.sqrt(coordinates[:, :1])
    weights = np.concatenate((1 - root, root * (1 - coordinates[:, 1:]),
                              root * coordinates[:, 1:]), axis=1)
    points = (mesh.triangles[faces] * weights[:, :, None]).sum(axis=1)
    normals = np.asarray(mesh.face_normals)[faces]
    return points.astype(np.float32), normals.astype(np.float32)


@torch.no_grad()
def sensor_surface_features(sensor_positions, object_positions, object_rotations,
                            surface_points, surface_normals, distance_scale):
    rotations = object_rotations[:, None, :].expand(-1, sensor_positions.shape[1], -1)
    local_sensors = quat_apply(
        quat_conjugate(rotations.reshape(-1, 4)),
        (sensor_positions - object_positions[:, None, :]).reshape(-1, 3),
    ).reshape_as(sensor_positions)
    distances, indices = torch.cdist(
        local_sensors, surface_points, compute_mode="donot_use_mm_for_euclid_dist",
    ).min(dim=-1)
    normals = surface_normals.gather(1, indices[:, :, None].expand(-1, -1, 3))
    world_normals = quat_apply(rotations.reshape(-1, 4), normals.reshape(-1, 3))
    features = torch.cat(((distances / distance_scale).clamp(0, 1)[:, :, None],
                          world_normals.reshape_as(sensor_positions)), dim=-1)
    return features.flatten(start_dim=1)
