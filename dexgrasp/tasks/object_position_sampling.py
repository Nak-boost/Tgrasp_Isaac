import math

import torch


def sample_hand_start_positions(count, x_range, y_range, inside_probability, ring_margin, device):
    bounds = torch.tensor([x_range, y_range], device=device, dtype=torch.float32)
    if not torch.isfinite(bounds).all() or not (bounds[:, 1] > bounds[:, 0]).all():
        raise ValueError('Hand initialization requires finite non-empty XY ranges')
    if not math.isfinite(inside_probability) or not 0 <= inside_probability <= 1:
        raise ValueError('Hand inside probability must be in [0, 1]')
    if not math.isfinite(ring_margin) or ring_margin <= 0:
        raise ValueError('Hand ring margin must be positive')
    random = torch.rand(count, 4, device=device)
    inside = random[:, 0] < inside_probability
    positions = bounds[:, 0] + random[:, 1:3] * (bounds[:, 1] - bounds[:, 0])
    lower = bounds[:, 0] - ring_margin
    upper = bounds[:, 1] + ring_margin
    lengths = upper - lower
    perimeter_position = random[:, 3] * (2 * lengths.sum())
    ring_x = torch.where(
        perimeter_position < lengths[0], lower[0] + perimeter_position,
        torch.where(perimeter_position < lengths.sum(), upper[0],
                    torch.where(perimeter_position < lengths.sum() + lengths[0],
                                upper[0] - (perimeter_position - lengths.sum()), lower[0])),
    )
    ring_y = torch.where(
        perimeter_position < lengths[0], lower[1],
        torch.where(perimeter_position < lengths.sum(), lower[1] + perimeter_position - lengths[0],
                    torch.where(perimeter_position < lengths.sum() + lengths[0], upper[1],
                                upper[1] - (perimeter_position - lengths.sum() - lengths[0]))),
    )
    ring_positions = torch.stack((ring_x, ring_y), dim=-1)
    return torch.where(inside.unsqueeze(-1), positions, ring_positions), ~inside


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
