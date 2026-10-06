#!/usr/bin/env python3

import argparse
import itertools
import os
import shutil
import subprocess
import sys
from datetime import datetime


def extract_visualization_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vis_steps", type=int, default=400)
    parser.add_argument("--vis_interval", type=int, default=1)
    parser.add_argument("--vis_output", type=str, default="voxel_vis")
    parser.add_argument("--vis_env", type=int, default=0)
    parser.add_argument("--vis_fps", type=int, default=20)
    parser.add_argument("--vis_unknown_limit", type=int, default=1500)
    parser.add_argument("--no_vis_video", action="store_true")
    visualization_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return visualization_args


VIS_ARGS = extract_visualization_args()

from utils.config import (  # noqa: E402
    get_args,
    load_cfg,
    parse_sim_params,
    set_np_formatting,
    set_seed,
)
from utils.parse_task import parse_task  # noqa: E402
from utils.process_marl import get_AgentIndex  # noqa: E402
from utils.process_sarl import process_ppo1  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def draw_workspace_box(axis, lower, upper):
    for varying_axis in range(3):
        fixed_axes = [axis for axis in range(3) if axis != varying_axis]
        for fixed_values in itertools.product((0, 1), repeat=2):
            start = lower.copy()
            end = lower.copy()
            end[varying_axis] = upper[varying_axis]
            for fixed_axis, fixed_value in zip(fixed_axes, fixed_values):
                coordinate = (
                    upper[fixed_axis] if fixed_value else lower[fixed_axis]
                )
                start[fixed_axis] = coordinate
                end[fixed_axis] = coordinate
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color="#777777",
                linewidth=0.6,
                alpha=0.5,
            )


def voxel_centers(lower, upper, grid_size):
    voxel_size = (upper - lower) / np.asarray(grid_size)
    coordinates = [
        lower[axis]
        + (np.arange(grid_size[axis]) + 0.5) * voxel_size[axis]
        for axis in range(3)
    ]
    return np.stack(
        np.meshgrid(*coordinates, indexing="ij"),
        axis=-1,
    )


def recency_colors(color, recency, minimum_alpha, maximum_alpha):
    colors = np.tile(np.asarray(to_rgba(color)), (len(recency), 1))
    colors[:, 3] = minimum_alpha + (
        maximum_alpha - minimum_alpha
    ) * np.clip(recency, 0.0, 1.0)
    return colors


def render_frame(
    output_path,
    step,
    reward,
    occupancy,
    recency,
    centers,
    lower,
    upper,
    expiration_threshold,
    sensor_positions,
    sensor_touch,
    estimated_center,
    estimate_quality,
    minimum_quality,
    true_center,
    table_top_z,
    unknown_limit,
):
    observed = recency >= expiration_threshold
    contact = observed & (occupancy > 0.5)
    free = observed & (np.abs(occupancy) <= 0.5)
    unknown = ~observed

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")

    unknown_indices = np.flatnonzero(unknown)
    if len(unknown_indices) > unknown_limit:
        stride = int(np.ceil(len(unknown_indices) / unknown_limit))
        unknown_indices = unknown_indices[::stride]
    unknown_points = centers.reshape(-1, 3)[unknown_indices]
    if len(unknown_points):
        axis.scatter(
            unknown_points[:, 0],
            unknown_points[:, 1],
            unknown_points[:, 2],
            color="#b8b8b8",
            marker=".",
            s=2,
            alpha=0.08,
            label="unknown (-1, sampled)",
        )

    free_points = centers[free]
    if len(free_points):
        axis.scatter(
            free_points[:, 0],
            free_points[:, 1],
            free_points[:, 2],
            c=recency_colors(
                "#2a9d8f", recency[free], 0.15, 0.65
            ),
            marker="s",
            s=8,
            linewidths=0,
            label="free (0)",
        )

    contact_points = centers[contact]
    if len(contact_points):
        axis.scatter(
            contact_points[:, 0],
            contact_points[:, 1],
            contact_points[:, 2],
            c=recency_colors(
                "#ff6b35", recency[contact], 0.45, 1.0
            ),
            marker="s",
            s=34,
            linewidths=0,
            label="contact (+1)",
        )

    sensor_colors = np.where(
        sensor_touch[:, None],
        np.asarray(to_rgba("#ff6b35")),
        np.asarray(to_rgba("#ffffff")),
    )
    axis.scatter(
        sensor_positions[:, 0],
        sensor_positions[:, 1],
        sensor_positions[:, 2],
        c=sensor_colors,
        marker="o",
        s=34,
        edgecolors="#111111",
        linewidths=0.7,
        label="tactile sensors",
    )

    estimate_valid = (
        estimate_quality >= minimum_quality
        and np.isfinite(estimated_center).all()
    )
    if estimate_valid:
        axis.scatter(
            *estimated_center,
            color="#d000ff",
            marker="*",
            s=180,
            edgecolors="#111111",
            linewidths=0.6,
            label="estimated center",
        )

    axis.scatter(
        *true_center,
        color="#70e000",
        marker="X",
        s=100,
        edgecolors="#111111",
        linewidths=0.6,
        label="true center (debug only)",
    )

    table_x, table_y = np.meshgrid(
        [lower[0], upper[0]],
        [lower[1], upper[1]],
    )
    axis.plot_surface(
        table_x,
        table_y,
        np.full_like(table_x, table_top_z),
        color="#777777",
        alpha=0.08,
        shade=False,
    )
    draw_workspace_box(axis, lower, upper)

    estimate_error = (
        np.linalg.norm(estimated_center - true_center) * 100.0
        if estimate_valid
        else float("nan")
    )
    estimate_status = (
        f"error={estimate_error:.1f} cm"
        if estimate_valid
        else "estimate invalid"
    )
    axis.set_title(
        f"step={step}  reward={reward:.3f}  "
        f"quality={estimate_quality:.3f}  {estimate_status}\n"
        f"unknown={unknown.sum()}  free={free.sum()}  "
        f"contact={contact.sum()}  active sensors={sensor_touch.sum()}"
    )
    axis.set_xlabel("world X (m)")
    axis.set_ylabel("world Y (m)")
    axis.set_zlabel("world Z (m)")
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])
    axis.set_box_aspect(upper - lower)
    axis.view_init(elev=27, azim=-58)
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def build_video(run_directory, fps):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("FFmpeg not found; PNG frames were saved without an MP4.")
        return

    video_path = os.path.join(run_directory, "voxel_visualization.mp4")
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(run_directory, "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            video_path,
        ],
        check=False,
    )
    if result.returncode == 0:
        print(f"Video saved to {video_path}")
    else:
        print(f"FFmpeg exited with status {result.returncode}.")


def main():
    if VIS_ARGS.vis_steps <= 0:
        raise ValueError("--vis_steps must be positive")
    if VIS_ARGS.vis_interval <= 0:
        raise ValueError("--vis_interval must be positive")
    if VIS_ARGS.vis_fps <= 0:
        raise ValueError("--vis_fps must be positive")
    if VIS_ARGS.vis_unknown_limit < 0:
        raise ValueError("--vis_unknown_limit must be non-negative")

    set_np_formatting()
    args = get_args()
    if not args.model_dir:
        raise ValueError("--model_dir must point to an E3 checkpoint")
    if not os.path.isfile(args.model_dir):
        raise FileNotFoundError(args.model_dir)

    args.test = True
    args.play = True
    args.train = False
    cfg, cfg_train, logdir = load_cfg(args)
    cfg["graphics_device_id"] = args.device_id
    cfg["env"]["random_time"] = False
    experiment = cfg["env"]["tactile"]["experiment"]
    if experiment != "E3":
        raise ValueError(
            "Voxel policy visualization requires an E3 environment config"
        )
    if cfg["env"]["numEnvs"] <= VIS_ARGS.vis_env:
        raise ValueError("--vis_env must be smaller than --num_envs")

    cfg_train["learn"]["test"] = True
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(
        cfg_train.get("seed", -1),
        cfg_train.get("torch_deterministic", False),
    )
    agent_index = get_AgentIndex(cfg)
    task, env = parse_task(
        args,
        cfg,
        cfg_train,
        sim_params,
        agent_index,
    )

    run_directory = os.path.join(
        os.path.abspath(VIS_ARGS.vis_output),
        datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_directory, exist_ok=False)
    ppo = process_ppo1(
        args,
        env,
        (cfg["env"], cfg_train),
        os.path.join(run_directory, "runtime"),
    )

    lower = task.voxel_lower.detach().cpu().numpy()
    upper = task.voxel_upper.detach().cpu().numpy()
    grid_size = tuple(task.voxel_grid_size)
    centers = voxel_centers(lower, upper, grid_size)
    expiration_threshold = task.voxel_expiration_threshold
    minimum_quality = task.tactile_position_minimum_quality
    table_top_z = float(task.table_top_z)
    env_id = VIS_ARGS.vis_env

    trace = {
        "step": [],
        "reward": [],
        "occupancy": [],
        "recency": [],
        "sensor_positions": [],
        "sensor_touch": [],
        "estimated_center": [],
        "estimate_quality": [],
        "true_center": [],
    }

    observations = env.reset()
    recurrent_hidden_states = ppo.initial_recurrent_state()
    print(
        f"Recording environment {env_id} from episode step 0 "
        f"for up to {VIS_ARGS.vis_steps} steps."
    )
    frame_index = 0
    for step in range(1, VIS_ARGS.vis_steps + 1):
        with torch.no_grad():
            actions, next_recurrent_hidden_states = ppo.act_inference(
                observations,
                recurrent_hidden_states,
            )
        results = env.step(actions)
        observations, rewards, dones = results[:3]
        recurrent_hidden_states = ppo.mask_recurrent_state(
            next_recurrent_hidden_states,
            dones,
        )
        episode_done = bool(dones[env_id].item())

        if step % VIS_ARGS.vis_interval != 0 and not episode_done:
            continue

        voxel = task.voxel_map[env_id].detach().cpu().numpy()
        occupancy = voxel[0]
        recency = voxel[1]
        sensor_positions = (
            task.touch_sensor_pos[env_id].detach().cpu().numpy()
        )
        sensor_touch = (
            task.binary_touch[env_id].detach().cpu().numpy() > 0.5
        )
        estimated_center = (
            task.tactile_object_position_estimate[env_id]
            .detach()
            .cpu()
            .numpy()
        )
        estimate_quality = float(
            task.tactile_object_position_quality[env_id].item()
        )
        true_center = (
            task.object_pos[env_id].detach().cpu().numpy()
        )
        reward = float(rewards[env_id].item())

        render_frame(
            os.path.join(
                run_directory, f"frame_{frame_index:04d}.png"
            ),
            step,
            reward,
            occupancy,
            recency,
            centers,
            lower,
            upper,
            expiration_threshold,
            sensor_positions,
            sensor_touch,
            estimated_center,
            estimate_quality,
            minimum_quality,
            true_center,
            table_top_z,
            VIS_ARGS.vis_unknown_limit,
        )

        trace["step"].append(step)
        trace["reward"].append(reward)
        trace["occupancy"].append(occupancy.astype(np.int8))
        trace["recency"].append(recency.astype(np.float16))
        trace["sensor_positions"].append(sensor_positions)
        trace["sensor_touch"].append(sensor_touch)
        trace["estimated_center"].append(estimated_center)
        trace["estimate_quality"].append(estimate_quality)
        trace["true_center"].append(true_center)
        frame_index += 1

        if episode_done:
            print(f"Episode ended at step {step}.")
            break

    np.savez_compressed(
        os.path.join(run_directory, "voxel_trace.npz"),
        **{key: np.asarray(values) for key, values in trace.items()},
        voxel_lower=lower,
        voxel_upper=upper,
        voxel_grid_size=np.asarray(grid_size),
        expiration_threshold=expiration_threshold,
    )
    ppo.writer.close()

    print(f"Saved {frame_index} frames to {run_directory}")
    if not VIS_ARGS.no_vis_video:
        build_video(run_directory, VIS_ARGS.vis_fps)


if __name__ == "__main__":
    main()
