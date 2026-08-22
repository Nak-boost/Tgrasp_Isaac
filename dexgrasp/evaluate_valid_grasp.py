import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from isaacgym import gymapi
import torch


DEXGRASP_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5")


def parse_evaluation_args():
    parser = argparse.ArgumentParser(
        description="Evaluate 0818 checkpoints with the current valid-grasp rule",
        add_help=False,
    )
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-num-envs", type=int, default=32)
    parser.add_argument("--checkpoint-iteration", type=int, default=10000)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--lift-threshold", type=float, default=0.01)
    parser.add_argument("--eval-root", default="logs/tactile_0818")
    parser.add_argument(
        "--output-dir",
        default="logs/tactile_0818/valid_grasp_evaluation",
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--single-experiment", choices=DEFAULT_EXPERIMENTS)
    parser.add_argument("-h", "--help", action="store_true")
    return parser.parse_known_args()


def print_help():
    print(
        """Evaluate the trained 0818 E1-E5 policies without changing training code.

Usage:
  python evaluate_valid_grasp.py [evaluation options] [Isaac Gym options]

Evaluation options:
  --experiments E1 E2 ...       Experiments to evaluate (default: E1-E5)
  --episodes N                  Completed episodes per experiment (default: 200)
  --eval-num-envs N             Parallel environments (default: 32)
  --checkpoint-iteration N      Preferred checkpoint (default: 10000)
  --eval-seed N                 Evaluation seed (default: 42)
  --lift-threshold METERS       Small-lift threshold (default: 0.01)
  --eval-root PATH              0818 log directory
  --output-dir PATH             CSV/JSON output directory
  --viewer                      Enable the Isaac Gym viewer

E5 automatically falls back to its latest checkpoint when model_10000.pt is
not available. Standard Isaac Gym options such as --sim_device cuda:1 and
--rl_device cuda:1 are passed through.
"""
    )


def resolve_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = DEXGRASP_ROOT / path
    return path.resolve()


def find_checkpoint(experiment_dir, preferred_iteration):
    preferred = experiment_dir / "checkpoint" / f"model_{preferred_iteration}.pt"
    if preferred.exists():
        return preferred, preferred_iteration

    checkpoints = []
    for checkpoint in (experiment_dir / "checkpoint").glob("model_*.pt"):
        try:
            iteration = int(checkpoint.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((iteration, checkpoint))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {experiment_dir / 'checkpoint'}")
    iteration, checkpoint = max(checkpoints, key=lambda item: item[0])
    return checkpoint, iteration


def run_all_experiments(eval_args, isaac_args):
    output_dir = resolve_path(eval_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for experiment in eval_args.experiments:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-experiment",
            experiment,
            "--episodes",
            str(eval_args.episodes),
            "--eval-num-envs",
            str(eval_args.eval_num_envs),
            "--checkpoint-iteration",
            str(eval_args.checkpoint_iteration),
            "--eval-seed",
            str(eval_args.eval_seed),
            "--lift-threshold",
            str(eval_args.lift_threshold),
            "--eval-root",
            str(resolve_path(eval_args.eval_root)),
            "--output-dir",
            str(output_dir),
        ]
        if eval_args.viewer:
            command.append("--viewer")
        command.extend(isaac_args)
        subprocess.run(command, cwd=DEXGRASP_ROOT, check=True)

        summary_path = output_dir / f"{experiment}_summary.json"
        with summary_path.open("r") as summary_file:
            summaries.append(json.load(summary_file))

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nCombined summary written to {summary_csv}")


def build_runtime_args(eval_args, isaac_args, experiment, checkpoint, config_path):
    sys.argv = [sys.argv[0], "--task", "ShadowHandGraspDexRep", "--algo", "ppo1", *isaac_args]

    from utils.config import get_args

    args = get_args()
    args.task = "ShadowHandGraspDexRep"
    args.task_type = "Python"
    args.algo = "ppo1"
    args.cfg_env = str(config_path)
    args.cfg_train = str(DEXGRASP_ROOT / "cfg" / "ppo1" / "config.yaml")
    args.logdir = str(resolve_path(eval_args.output_dir) / experiment)
    args.model_dir = str(checkpoint)
    args.test = True
    args.play = True
    args.train = False
    args.headless = not eval_args.viewer
    args.num_envs = eval_args.eval_num_envs
    args.seed = eval_args.eval_seed
    return args


def append_terminal_episodes(records, terminal_ids, state, infos, limit):
    remaining = limit - len(records)
    for env_id in terminal_ids[:remaining].tolist():
        records.append(
            {
                "episode": len(records) + 1,
                "valid_grasp": int(state["ever_valid"][env_id].item()),
                "first_valid_grasp_step": float(
                    state["first_valid_step"][env_id].item()
                ),
                "valid_grasp_steps": int(state["valid_steps"][env_id].item()),
                "max_valid_grasp_run": int(state["max_valid_run"][env_id].item()),
                "max_lift_after_valid_grasp_m": float(
                    state["max_lift_after_valid"][env_id].item()
                ),
                "max_lift_while_valid_grasp_m": float(
                    state["max_lift_while_valid"][env_id].item()
                ),
                "target_lift_while_valid_grasp": int(
                    state["target_lift_while_valid"][env_id].item()
                ),
                "episode_steps": int(state["episode_steps"][env_id].item()),
                "environment_success": float(
                    infos["successes"].reshape(-1)[env_id].item()
                ),
            }
        )


def reset_episode_state(state, terminal_ids):
    for name, values in state.items():
        if name == "first_valid_step":
            values[terminal_ids] = -1
        else:
            values[terminal_ids] = 0


def summarize_records(experiment, checkpoint_iteration, records, lift_threshold):
    valid_records = [record for record in records if record["valid_grasp"]]
    lifted_records = [
        record
        for record in valid_records
        if record["max_lift_after_valid_grasp_m"] >= lift_threshold
    ]
    target_records = [
        record for record in valid_records if record["target_lift_while_valid_grasp"]
    ]

    def mean(key, selected_records):
        if not selected_records:
            return 0.0
        return statistics.mean(record[key] for record in selected_records)

    episode_count = len(records)
    valid_count = len(valid_records)
    return {
        "experiment": experiment,
        "checkpoint_iteration": checkpoint_iteration,
        "episodes": episode_count,
        "valid_grasp_episode_rate": valid_count / episode_count,
        "mean_first_valid_grasp_step": mean(
            "first_valid_grasp_step", valid_records
        ),
        "mean_valid_grasp_steps": mean("valid_grasp_steps", valid_records),
        "mean_max_valid_grasp_run": mean("max_valid_grasp_run", valid_records),
        "lift_after_valid_grasp_rate": (
            len(lifted_records) / valid_count if valid_count else 0.0
        ),
        "target_lift_while_valid_grasp_rate": (
            len(target_records) / valid_count if valid_count else 0.0
        ),
        "mean_max_lift_after_valid_grasp_m": mean(
            "max_lift_after_valid_grasp_m", valid_records
        ),
        "environment_success_rate": mean("environment_success", records),
    }


def evaluate_single(eval_args, isaac_args):
    experiment = eval_args.single_experiment
    eval_root = resolve_path(eval_args.eval_root)
    experiment_dir = eval_root / f"{experiment}_seed42"
    config_path = experiment_dir / f"shadow_hand_grasp_dexrep_{experiment}.yaml"
    checkpoint, checkpoint_iteration = find_checkpoint(
        experiment_dir, eval_args.checkpoint_iteration
    )
    if checkpoint_iteration != eval_args.checkpoint_iteration:
        print(
            f"{experiment}: model_{eval_args.checkpoint_iteration}.pt is unavailable; "
            f"using model_{checkpoint_iteration}.pt"
        )
    if not config_path.exists():
        raise FileNotFoundError(f"Saved environment config not found: {config_path}")

    args = build_runtime_args(
        eval_args, isaac_args, experiment, checkpoint, config_path
    )

    from utils.config import load_cfg, parse_sim_params, set_np_formatting, set_seed
    from utils.parse_task import parse_task
    from utils.process_marl import get_AgentIndex
    from utils.process_sarl import process_ppo1

    set_np_formatting()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(eval_args.eval_seed, cfg_train.get("torch_deterministic", False))
    task, env = parse_task(
        args, cfg, cfg_train, sim_params, get_AgentIndex(cfg)
    )
    model = process_ppo1(args, env, (cfg["env"], cfg_train), logdir)

    device = torch.device(args.rl_device)
    num_envs = env.num_envs
    state = {
        "ever_valid": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "first_valid_step": torch.full(
            (num_envs,), -1.0, dtype=torch.float, device=device
        ),
        "valid_steps": torch.zeros(num_envs, dtype=torch.long, device=device),
        "current_valid_run": torch.zeros(
            num_envs, dtype=torch.long, device=device
        ),
        "max_valid_run": torch.zeros(num_envs, dtype=torch.long, device=device),
        "max_lift_after_valid": torch.zeros(
            num_envs, dtype=torch.float, device=device
        ),
        "max_lift_while_valid": torch.zeros(
            num_envs, dtype=torch.float, device=device
        ),
        "target_lift_while_valid": torch.zeros(
            num_envs, dtype=torch.bool, device=device
        ),
        "episode_steps": torch.zeros(num_envs, dtype=torch.long, device=device),
    }

    records = []
    observations = env.reset()
    while len(records) < eval_args.episodes:
        with torch.no_grad():
            actions = model.actor_critic.act_inference(observations)
            observations, _, dones, infos = env.step(actions)

        valid_grasp = infos["upward_action_supervision_mask"].reshape(-1) > 0.5
        state["episode_steps"].add_(1)
        newly_valid = torch.logical_and(valid_grasp, ~state["ever_valid"])
        state["first_valid_step"] = torch.where(
            newly_valid,
            state["episode_steps"].float(),
            state["first_valid_step"],
        )
        state["ever_valid"].logical_or_(valid_grasp)
        state["valid_steps"].add_(valid_grasp.long())
        state["current_valid_run"] = torch.where(
            valid_grasp,
            state["current_valid_run"] + 1,
            torch.zeros_like(state["current_valid_run"]),
        )
        state["max_valid_run"] = torch.maximum(
            state["max_valid_run"], state["current_valid_run"]
        )

        lift_amount = torch.clamp(
            task.object_pos[:, 2] - task.episode_object_start_height,
            min=0.0,
        ).to(device)
        state["max_lift_after_valid"] = torch.where(
            state["ever_valid"],
            torch.maximum(state["max_lift_after_valid"], lift_amount),
            state["max_lift_after_valid"],
        )
        state["max_lift_while_valid"] = torch.where(
            valid_grasp,
            torch.maximum(state["max_lift_while_valid"], lift_amount),
            state["max_lift_while_valid"],
        )
        state["target_lift_while_valid"].logical_or_(
            valid_grasp & (lift_amount >= task.touch_lift_target_height)
        )

        terminal_ids = torch.nonzero(dones.reshape(-1) > 0, as_tuple=False).flatten()
        if terminal_ids.numel() > 0:
            append_terminal_episodes(
                records, terminal_ids, state, infos, eval_args.episodes
            )
            reset_episode_state(state, terminal_ids)
            print(
                f"\r{experiment}: {len(records)}/{eval_args.episodes} episodes",
                end="",
                flush=True,
            )

    print()
    summary = summarize_records(
        experiment, checkpoint_iteration, records, eval_args.lift_threshold
    )
    output_dir = resolve_path(eval_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_csv = output_dir / f"{experiment}_episodes.csv"
    with episode_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / f"{experiment}_summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(json.dumps(summary, indent=2))
    model.writer.close()


def main():
    os.chdir(DEXGRASP_ROOT)
    eval_args, isaac_args = parse_evaluation_args()
    if eval_args.help:
        print_help()
        return
    if eval_args.episodes < 1 or eval_args.eval_num_envs < 1:
        raise ValueError("--episodes and --eval-num-envs must be positive")
    if eval_args.single_experiment:
        evaluate_single(eval_args, isaac_args)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        run_all_experiments(eval_args, isaac_args)


if __name__ == "__main__":
    main()
