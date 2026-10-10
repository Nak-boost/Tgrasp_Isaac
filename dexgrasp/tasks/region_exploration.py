import math

import torch


class RegionExplorationReward:
    def __init__(
        self, num_envs, grid_size, voxel_lower, voxel_upper,
        search_lower, search_upper, reference_height, height_layers=3,
        coverage_threshold=0.7, scale=1.0,
    ):
        if height_layers not in (2, 3) or height_layers > grid_size[2]:
            raise ValueError("Region exploration requires two or three height layers")
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("Region exploration scale must be finite and non-negative")
        if not math.isfinite(coverage_threshold) or not 0 < coverage_threshold <= 1:
            raise ValueError("Region exploration coverage_threshold must be in (0, 1]")
        if not torch.isfinite(search_lower).all() or not torch.isfinite(search_upper).all():
            raise ValueError("Region exploration search bounds must be finite")
        if not (search_upper > search_lower).all():
            raise ValueError("Region exploration requires a non-empty XY search range")
        if (search_lower < voxel_lower[:2]).any() or (search_upper > voxel_upper[:2]).any():
            raise ValueError("Object randomization range must lie inside the voxel map")
        if (
            not math.isfinite(reference_height)
            or not voxel_lower[2] <= reference_height < voxel_upper[2]
        ):
            raise ValueError("Region exploration reference height must lie inside the voxel map")

        self.scale = scale
        self.coverage_threshold = coverage_threshold
        device = voxel_lower.device
        cell_size = (voxel_upper - voxel_lower) / voxel_lower.new_tensor(grid_size)
        layer_start = round(
            float((reference_height - voxel_lower[2]) / cell_size[2])
            - height_layers / 2
        )
        layer_start = max(0, min(layer_start, grid_size[2] - height_layers))
        self.height_slice = slice(layer_start, layer_start + height_layers)

        region_axes = []
        overlaps = []
        for axis in range(2):
            region_edges = (
                search_lower[axis]
                + torch.arange(4, device=device)
                * (search_upper[axis] - search_lower[axis]) / 3
            )
            cell_edges = (
                voxel_lower[axis]
                + torch.arange(grid_size[axis] + 1, device=device) * cell_size[axis]
            )
            overlaps.append((
                torch.minimum(cell_edges[1:, None], region_edges[None, 1:])
                - torch.maximum(cell_edges[:-1, None], region_edges[None, :-1])
            ).clamp(min=0))
            region_axes.append((region_edges[:-1] + region_edges[1:]) / 2)

        weights = (
            overlaps[0][:, None, :, None] * overlaps[1][None, :, None, :]
        ).reshape(-1, 9)
        self.region_weights = weights / weights.sum(dim=0, keepdim=True)
        self.centers = torch.stack(
            torch.meshgrid(*region_axes, indexing="ij"), dim=-1
        ).reshape(9, 2)
        self.region_half_size = (search_upper - search_lower) / 6
        self.coverage = torch.zeros(num_envs, 9, device=device)
        self.explored_regions = torch.zeros(num_envs, 9, dtype=torch.bool, device=device)
        self.target = torch.zeros(num_envs, 2, device=device)
        self.target_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.previous_distance = torch.zeros(num_envs, device=device)

    def distance_to_region(self, palm_xy, centers):
        offsets = (torch.abs(palm_xy - centers) - self.region_half_size).clamp(min=0)
        return torch.norm(offsets, dim=-1)

    @torch.no_grad()
    def update(self, occupancy, palm_xy, search_active):
        current_distance = self.distance_to_region(palm_xy, self.target)
        reward = torch.where(
            self.target_valid & search_active,
            self.scale * (self.previous_distance - current_distance),
            torch.zeros_like(current_distance),
        )

        observed_xy = (occupancy[..., self.height_slice] > -0.5).any(dim=-1)
        self.coverage.copy_(
            (observed_xy.flatten(1).float() @ self.region_weights).clamp(0, 1)
        )
        self.explored_regions.logical_or_(
            self.coverage >= self.coverage_threshold - 1.0e-6
        )
        candidates = ~self.explored_regions & search_active.unsqueeze(1)
        distances = self.distance_to_region(
            palm_xy.unsqueeze(1), self.centers.unsqueeze(0)
        )
        target_index = distances.masked_fill(~candidates, float("inf")).argmin(dim=1)
        self.target.copy_(self.centers[target_index])
        self.previous_distance.copy_(
            distances.gather(1, target_index.unsqueeze(1)).squeeze(1)
        )
        self.target_valid.copy_(candidates.any(dim=1))
        return reward

    def reset(self, env_ids):
        self.coverage[env_ids] = 0
        self.explored_regions[env_ids] = False
        self.target[env_ids] = 0
        self.target_valid[env_ids] = False
        self.previous_distance[env_ids] = 0
