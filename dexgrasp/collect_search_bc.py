import argparse
import json
import os
import sys
from pathlib import Path


def collection_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bc_output", required=True)
    parser.add_argument("--bc_steps", type=int, default=0)
    parser.add_argument("--bc_speed", type=float, default=0.075)
    parser.add_argument("--bc_lane_spacing", type=float, default=0.10)
    parser.add_argument("--bc_sensor_height", type=float, default=0.64)
    parser.add_argument("--bc_kp", type=float, default=4000.0)
    parser.add_argument("--bc_kd", type=float, default=80.0)
    parser.add_argument("--bc_drag", type=float, default=2650.0)
    parser.add_argument("--bc_finger_noise", type=float, default=0.025)
    parser.add_argument("--bc_with_objects", action="store_true")
    parser.add_argument("--bc_checkpoint", default="")
    parsed, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return parsed


OPTIONS = collection_arguments()

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from utils.process_marl import get_AgentIndex
from utils.search_demonstration import BasePathController, sample_path, serpentine_waypoints
from isaacgym import gymtorch

import numpy as np
import torch
import yaml


def save_plot(output, traces, bounds, table_height):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for trace in traces:
        axes[0].plot(trace["base_state"][:, 0], trace["base_state"][:, 1], alpha=0.6)
    trace = traces[0]
    axes[0].plot(trace["target_position"][:, 0], trace["target_position"][:, 1], "k--", linewidth=1)
    axes[0].set(title="Palm body XY: actual / planned", xlabel="x (m)", ylabel="y (m)")
    axes[0].set_aspect("equal")
    for sensor in range(trace["sensor_positions"].shape[1]):
        axes[1].plot(trace["sensor_positions"][:, sensor, 0], trace["sensor_positions"][:, sensor, 1], linewidth=0.7)
    xmin, xmax, ymin, ymax = bounds
    axes[1].plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], "k--")
    axes[1].set(title="Tactile sensor XY trajectories", xlabel="x (m)", ylabel="y (m)")
    axes[1].set_aspect("equal")
    axes[2].plot(trace["sensor_positions"][:, :, 2])
    axes[2].axhline(table_height, color="black")
    axes[2].axhline(table_height + 0.015, color="red", linestyle="--")
    axes[2].set(title="Sensor heights / table / touch cutoff", xlabel="step", ylabel="z (m)")
    figure.tight_layout()
    figure.savefig(output / "trajectory.png", dpi=150)
    plt.close(figure)


def main():
    args = get_args()
    cfg, cfg_train, _ = load_cfg(args)
    if cfg["env"]["tactile"]["experiment"] != "E3":
        raise ValueError("Search demonstrations currently require E3")
    if cfg["env"].get("useRelativeControl", False):
        raise ValueError("Search demonstrations require base force control")
    output = Path(OPTIONS.bc_output)
    output.mkdir(parents=True, exist_ok=False)
    cfg["env"]["random_time"] = False
    cfg["task"]["randomize"] = False
    cfg_train["seed"] = set_seed(cfg_train.get("seed", 42))
    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params, get_AgentIndex(cfg))
    device = task.device
    observations = env.reset()
    if not OPTIONS.bc_with_objects:
        object_indices = task.object_indices.to(torch.int32)
        task.root_state_tensor[task.object_indices, :3] = torch.tensor([2.0, 2.0, 0.7], device=device)
        task.root_state_tensor[task.object_indices, 7:13] = 0
        task.gym.set_actor_root_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task.root_state_tensor), gymtorch.unwrap_tensor(object_indices), len(object_indices))
    controller = BasePathController(task, OPTIONS.bc_kp, OPTIONS.bc_kd, OPTIONS.bc_drag)
    palm_index = task.hand_body_idx_dict["palm"]
    initial_state = task.rigid_body_states[:, palm_index].clone()
    orientation = initial_state[:, 3:7].clone()
    lower = task.shadow_hand_dof_lower_limits[task.actuated_dof_indices]
    upper = task.shadow_hand_dof_upper_limits[task.actuated_dof_indices]
    finger_positions = task.shadow_hand_dof_pos[:, task.actuated_dof_indices].clone()
    finger_positions += OPTIONS.bc_finger_noise * (2 * torch.rand_like(finger_positions) - 1)
    finger_positions = torch.max(torch.min(finger_positions, upper), lower)
    finger_actions = 2 * (finger_positions - lower) / (upper - lower) - 1
    records = {name: [] for name in ("observations", "actions", "base_state", "target_position", "sensor_positions", "bc_mask", "next_contact", "tracking_error", "orientation_error", "region_coverage")}
    contact_seen = task.binary_touch.any(dim=-1)
    policy = None
    hidden = None
    if OPTIONS.bc_checkpoint:
        from algorithms.rl.ppo1.multimodal_policy import ActorCriticMultimodal
        policy = ActorCriticMultimodal(env.observation_space.shape, env.action_space.shape, cfg_train["learn"]["init_noise_std"], cfg_train["policy"], cfg_train["encoder"], cfg["env"]).to(device)
        policy.load_state_dict(torch.load(OPTIONS.bc_checkpoint, map_location=device))
        policy.eval()
        hidden = policy.initial_recurrent_state(task.num_envs, device)

    def advance(target, velocity):
        nonlocal observations, hidden, contact_seen
        actions = controller.actions(target, velocity, orientation, finger_actions)
        if policy is not None:
            with torch.no_grad():
                prediction = policy.act_inference(observations, hidden)
                if policy.is_recurrent:
                    prediction, hidden = prediction
                actions[:, :6] = prediction[:, :6].clamp(-1, 1)
        records["observations"].append(observations.detach().cpu().to(torch.float16).numpy().copy())
        records["actions"].append(actions.detach().cpu().numpy().copy())
        records["base_state"].append(task.rigid_body_states[:, palm_index].cpu().numpy().copy())
        records["target_position"].append(target.cpu().numpy())
        records["sensor_positions"].append(task.touch_sensor_pos.cpu().numpy().copy())
        records["bc_mask"].append((~contact_seen).cpu().numpy())
        observations, _, done, _ = env.step(actions)
        contact = task.binary_touch.any(dim=-1)
        contact_seen |= contact
        error = (task.rigid_body_states[:, palm_index, :3] - target).norm(dim=-1)
        angle = 2 * torch.acos((task.rigid_body_states[:, palm_index, 3:7] * orientation).sum(dim=-1).abs().clamp(0, 1))
        records["next_contact"].append(contact.cpu().numpy())
        records["tracking_error"].append(error.cpu().numpy())
        records["orientation_error"].append(angle.cpu().numpy())
        coverage = task.region_exploration.coverage if task.region_exploration is not None else torch.zeros(task.num_envs, 9, device=device)
        records["region_coverage"].append(coverage.cpu().numpy().copy())
        if done.any():
            raise RuntimeError("Unexpected episode reset during collection; inspect contacts and horizon")
        return error, coverage

    for _ in range(60):
        advance(initial_state[:, :3], torch.zeros_like(initial_state[:, :3]))
    palm_state = task.rigid_body_states[:, palm_index].clone()
    sensor_positions = task.touch_sensor_pos.clone()
    print("Initial palm:", palm_state[0].tolist(), flush=True)
    print("Settled sensors:", sensor_positions[0].tolist(), flush=True)
    print("Finger DOFs:", task.shadow_hand_dof_pos[0].tolist(), flush=True)
    print("Object locations (diagnostic only):", task.object_pos.tolist(), flush=True)
    randomization = cfg["env"]["objectPositionRandomization"]
    bounds = [*randomization["xRange"], *randomization["yRange"]]
    dt = task.dt * task.control_freq_inv
    paths, velocities, waypoints = [], [], []
    for env_index in range(task.num_envs):
        offset = (sensor_positions[env_index].mean(dim=0) - palm_state[env_index, :3]).cpu().numpy()
        points = serpentine_waypoints(palm_state[env_index, :3].cpu().numpy(), offset, bounds, OPTIONS.bc_sensor_height, OPTIONS.bc_lane_spacing)
        speed = OPTIONS.bc_speed * (1 + 0.04 * (env_index / max(task.num_envs - 1, 1) - 0.5))
        positions, rates = sample_path(points, speed, dt)
        paths.append(torch.as_tensor(positions, device=device))
        velocities.append(torch.as_tensor(rates, device=device))
        waypoints.append(points.tolist())
    required_steps = max(len(path) for path in paths)
    steps = min(OPTIONS.bc_steps, required_steps) if OPTIONS.bc_steps else required_steps
    task.max_episode_length = max(task.max_episode_length, steps + 70)
    for step in range(steps):
        target = torch.stack([path[min(step, len(path) - 1)] for path in paths])
        velocity = torch.stack([rates[min(step, len(rates) - 1)] if step < len(rates) else torch.zeros(3, device=device) for rates in velocities])
        error, coverage = advance(target, velocity)
        if step % 200 == 0:
            print(f"step={step}/{steps} mean_error={error.mean().item():.4f} contact_envs={contact_seen.sum().item()} mean_coverage={coverage.mean().item():.3f}", flush=True)
    arrays = {name: np.stack(values) for name, values in records.items()}
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise RuntimeError("Non-finite trajectory; dataset was not written")
    traces, summaries = [], []
    for env_index in range(task.num_envs):
        length = 60 + min(steps, len(paths[env_index]))
        trace = {name: values[:length, env_index] for name, values in arrays.items()}
        np.savez_compressed(output / f"trajectory_{env_index:04d}.npz", **trace)
        traces.append(trace)
        summaries.append({
            "trajectory": env_index, "steps": length,
            "bc_samples": int(trace["bc_mask"].sum()),
            "complete": steps >= len(paths[env_index]),
            "tracking_rmse_m": float(np.sqrt(np.mean(trace["tracking_error"] ** 2))),
            "tracking_p95_m": float(np.percentile(trace["tracking_error"], 95)),
            "orientation_max_deg": float(np.rad2deg(trace["orientation_error"].max())),
            "contact_steps": int(trace["next_contact"].sum()),
            "base_saturation_fraction": float((np.abs(trace["actions"][:, :6]) >= 0.999).mean()),
            "final_region_coverage": trace["region_coverage"][-1].tolist(),
        })
    metadata = {"schema_version": 1, "collection": vars(OPTIONS), "args": vars(args), "env": cfg["env"], "train": cfg_train, "dt": dt, "action_indices": list(range(6)), "waypoints": waypoints, "summaries": summaries}
    with (output / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    with (output / "resolved_config.yaml").open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    save_plot(output, traces, bounds, task.table_top_z)
    print(json.dumps(summaries, indent=2), flush=True)
    torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
