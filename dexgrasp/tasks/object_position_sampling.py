import math

import torch


def sample_inner_outer_offsets(count, x_range, y_range, fraction, probability, device):
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('Inner region half-extent fraction must be in (0, 1)')
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Inner region probability must be in [0, 1]')
    bounds = torch.tensor([x_range, y_range], device=device, dtype=torch.float32)
    if not torch.isfinite(bounds).all() or not (bounds[:, 1] > bounds[:, 0]).all():
        raise ValueError('Position sampling requires finite non-empty XY ranges')
    random = torch.rand(count, 4, device=device)
    inner = random[:, 0] < probability
    radius_squared = torch.where(
        inner, random[:, 1] * fraction ** 2,
        fraction ** 2 + random[:, 1] * (1 - fraction ** 2),
    )
    radius = radius_squared.sqrt()
    side = (random[:, 2] * 4).long()
    normal = torch.where(side % 2 == 0, radius, -radius)
    tangent = (2 * random[:, 3] - 1) * radius
    normalized = torch.stack((
        torch.where(side < 2, normal, tangent),
        torch.where(side < 2, tangent, normal),
    ), dim=-1)
    return bounds.mean(dim=1) + normalized * (bounds[:, 1] - bounds[:, 0]) / 2
