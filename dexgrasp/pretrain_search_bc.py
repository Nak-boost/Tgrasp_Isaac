import argparse
import json
from pathlib import Path

from isaacgym import gymapi

import numpy as np
import torch

from algorithms.rl.ppo1.multimodal_policy import ActorCriticMultimodal

PHASE_NAMES = ("settle", "descend", "approach", "sweep", "turn")


def sequence_predictions(model, observations, hidden, recurrent=None):
    steps, batch, dimensions = observations.shape
    embeddings = model.encode_observations(observations.reshape(steps * batch, dimensions))
    embeddings = embeddings.reshape(steps, batch, -1)
    if model.is_recurrent:
        if recurrent is not None:
            recurrent.train(model.training)
            states, final_state = recurrent(embeddings, hidden.unsqueeze(0))
            hidden = final_state.squeeze(0)
            features = model.recurrent_normalizer(states)
        else:
            states = []
            for embedding in embeddings:
                hidden = model.recurrent_memory(embedding, hidden)
                states.append(hidden)
            features = model.recurrent_normalizer(torch.stack(states))
    else:
        features = embeddings
    predictions = model.actor(features.reshape(steps * batch, -1))[:, :6]
    return predictions.reshape(steps, batch, 6), hidden


def trajectory_pass(model, files, device, sequence_length, optimizer=None, cache=None, recurrent=None):
    model.train(optimizer is not None)
    total_squared_error = np.zeros(6, dtype=np.float64)
    total_samples = 0
    phase_error = np.zeros((5, 6), dtype=np.float64)
    phase_count = np.zeros(5, dtype=np.int64)
    for filename in files:
        archive = cache[filename] if cache is not None else np.load(filename)
        observations = archive["observations"]
        labels = archive.get("expert_actions", archive["actions"])[:, :6]
        valid = archive["bc_mask"].astype(bool)
        phases = archive["phase"] if "phase" in archive else np.full(len(valid), 3)
        weights = np.zeros(len(valid), dtype=np.float32)
        for phase_id, importance in enumerate((0.25, 2.0, 1.0, 1.0, 1.0)):
            selected = (phases == phase_id) & valid
            weights[selected] = importance / max(int(selected.sum()), 1)
        weights /= max(float(weights.sum()), 1e-8)
        if not np.isfinite(observations).all() or not np.isfinite(labels).all():
            raise ValueError(f"Non-finite demonstration: {filename}")
        hidden = model.initial_recurrent_state(1, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(observations), sequence_length):
            stop = start + sequence_length
            batch = torch.as_tensor(observations[start:stop], device=device, dtype=torch.float32).unsqueeze(1)
            target = torch.as_tensor(labels[start:stop], device=device).unsqueeze(1)
            mask = torch.as_tensor(valid[start:stop], device=device).unsqueeze(1)
            with torch.set_grad_enabled(optimizer is not None):
                prediction, hidden = sequence_predictions(model, batch, hidden, recurrent)
                errors = (prediction - target).square()[mask]
                if len(errors):
                    loss = errors.mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite BC loss")
                    if optimizer is not None:
                        weight = torch.as_tensor(weights[start:stop][valid[start:stop]], device=device)
                        (errors.mean(dim=-1) * weight).sum().backward()
                    total_squared_error += errors.detach().sum(dim=0).cpu().numpy()
                    total_samples += len(errors)
                    error_numpy = errors.detach().cpu().numpy()
                    for phase_id in range(5):
                        selected = phases[start:stop][valid[start:stop]] == phase_id
                        phase_error[phase_id] += error_numpy[selected].sum(axis=0)
                        phase_count[phase_id] += selected.sum()
            if hidden is not None:
                hidden = hidden.detach()
        if optimizer is not None and valid.any():
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    if total_samples == 0:
        raise ValueError("No eligible pre-contact BC samples")
    phase_metrics = {name: {"mse": float(phase_error[index].mean() / max(phase_count[index], 1)), "samples": int(phase_count[index])} for index, name in enumerate(PHASE_NAMES)}
    balanced_mse = np.mean([entry["mse"] for entry in phase_metrics.values() if entry["samples"]])
    return {"mse": float(total_squared_error.mean() / total_samples), "balanced_mse": float(balanced_mse), "axis_rmse": np.sqrt(total_squared_error / total_samples).tolist(), "samples": total_samples, "phases": phase_metrics}


def main():
    parser = argparse.ArgumentParser(description="Pretrain only E3 base action means on continuous search demonstrations")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--replay_dataset", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--base_std", type=float, default=0.05)
    parser.add_argument("--finger_std", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial_checkpoint", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.sequence_length, args.learning_rate, args.save_every) <= 0:
        raise ValueError("Epochs, sequence length and learning rate must be positive")
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    with (args.dataset / "metadata.json").open() as handle:
        metadata = json.load(handle)
    cache = {}
    training_files, validation_files = [], []
    datasets = [args.dataset, *args.replay_dataset]
    if len(set(directory.resolve() for directory in datasets)) != len(datasets):
        raise ValueError("Dataset directories must be distinct")
    for directory in datasets:
        with (directory / "metadata.json").open() as handle:
            source = json.load(handle)
        if source.get("schema_version") != 1 or source.get("action_indices") != list(range(6)):
            raise ValueError("Unsupported search dataset schema")
        if list(source["env"]["obs_dim"].items()) != list(metadata["env"]["obs_dim"].items()):
            raise ValueError("Replay datasets must use the same observation layout")
        for setting in ("include_base_velocity", "center_base_euler"):
            if source["env"]["tactile"].get(setting, False) != metadata["env"]["tactile"].get(setting, False):
                raise ValueError("Replay datasets must use the same base observation encoding")
        collection = source["collection"]
        if collection.get("bc_checkpoint") and not collection.get("bc_expert_blend", 0):
            raise ValueError("Policy evaluation trajectories are not expert BC labels")
        files = sorted(directory.glob("trajectory_*.npz"))
        if len(files) < 2:
            raise ValueError("Each dataset requires at least two trajectories")
        training_files.extend(files[:-1])
        validation_files.extend(files[-1:])
        for filename in files:
            with np.load(filename) as archive:
                cache[filename] = {name: archive[name] for name in ("observations", "actions", "expert_actions", "bc_mask", "phase") if name in archive}
                if (collection.get("bc_action_noise", 0) or collection.get("bc_expert_blend", 0)) and "expert_actions" not in cache[filename]:
                    raise ValueError("Corrective demonstrations require clean expert action labels")
    args.output.mkdir(parents=True, exist_ok=False)
    env_cfg, train_cfg = metadata["env"], metadata["train"]
    model = ActorCriticMultimodal(
        (sum(env_cfg["obs_dim"].values()),), (24,),
        train_cfg["learn"]["init_noise_std"], train_cfg["policy"], train_cfg["encoder"], env_cfg,
    ).to(args.device)
    if args.initial_checkpoint:
        model.load_state_dict(torch.load(args.initial_checkpoint, map_location=args.device))
    else:
        reference = np.mean([cache[filename]["actions"][0, 6:] for filename in training_files], axis=0)
        with torch.no_grad():
            model.actor[-1].weight[6:].zero_()
            model.actor[-1].bias[6:].copy_(torch.as_tensor(reference, device=args.device))
            if not model.minimum_action_std <= min(args.base_std, args.finger_std) <= max(args.base_std, args.finger_std) <= model.maximum_action_std:
                raise ValueError("Initial action standard deviations must satisfy policy bounds")
            if model.state_dependent_std:
                model.log_std_head.weight.zero_()
                model.log_std_head.bias[:6].fill_(np.log(args.base_std))
                model.log_std_head.bias[6:].fill_(np.log(args.finger_std))
            else:
                model.log_std[:6].fill_(np.log(args.base_std))
                model.log_std[6:].fill_(np.log(args.finger_std))
    for name, parameter in model.named_parameters():
        if name.startswith(("critic.", "log_std_head.")) or name == "log_std":
            parameter.requires_grad_(False)
    recurrent = None
    if model.is_recurrent:
        cell = model.recurrent_memory
        recurrent = torch.nn.GRU(cell.input_size, cell.hidden_size).to(args.device)
        recurrent.weight_ih_l0 = cell.weight_ih
        recurrent.weight_hh_l0 = cell.weight_hh
        recurrent.bias_ih_l0 = cell.bias_ih
        recurrent.bias_hh_l0 = cell.bias_hh
        recurrent.flatten_parameters()
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    metrics = []
    best_loss = float("inf")
    initial = trajectory_pass(model, validation_files, args.device, args.sequence_length, cache=cache, recurrent=recurrent)
    print("Initial validation:", json.dumps(initial), flush=True)
    for epoch in range(args.epochs):
        generator.shuffle(training_files)
        training = trajectory_pass(model, training_files, args.device, args.sequence_length, optimizer, cache, recurrent)
        validation = trajectory_pass(model, validation_files, args.device, args.sequence_length, cache=cache, recurrent=recurrent)
        entry = {"epoch": epoch + 1, "training": training, "validation": validation}
        metrics.append(entry)
        print(json.dumps(entry), flush=True)
        if validation["balanced_mse"] < best_loss:
            best_loss = validation["balanced_mse"]
            torch.save(model.state_dict(), args.output / "model_0.pt")
        if (epoch + 1) % args.save_every == 0:
            checkpoint = args.output / f"epoch_{epoch + 1:03d}"
            checkpoint.mkdir()
            torch.save(model.state_dict(), checkpoint / "model_0.pt")
        with (args.output / "metrics.json").open("w") as handle:
            json.dump({"initial_validation": initial, "epochs": metrics}, handle, indent=2)
    with (args.output / "pretraining.json").open("w") as handle:
        json.dump({"args": vars(args), "env": env_cfg, "train": train_cfg, "training_files": training_files, "validation_files": validation_files, "loss_action_indices": list(range(6))}, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
