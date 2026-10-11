import argparse
import json
from pathlib import Path

from isaacgym import gymapi

import numpy as np
import torch

from algorithms.rl.ppo1.multimodal_policy import ActorCriticMultimodal


def sequence_predictions(model, observations, hidden):
    steps, batch, dimensions = observations.shape
    embeddings = model.encode_observations(observations.reshape(steps * batch, dimensions))
    embeddings = embeddings.reshape(steps, batch, -1)
    predictions = []
    for embedding in embeddings:
        features, hidden = model.apply_recurrent_memory(embedding, hidden)
        predictions.append(model.actor(features)[:, :6])
    return torch.stack(predictions), hidden


def trajectory_pass(model, files, device, sequence_length, optimizer=None):
    model.train(optimizer is not None)
    total_squared_error = np.zeros(6, dtype=np.float64)
    total_samples = 0
    for filename in files:
        with np.load(filename) as archive:
            observations = archive["observations"]
            labels = archive["actions"][:, :6]
            valid = archive["bc_mask"].astype(bool)
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
                prediction, hidden = sequence_predictions(model, batch, hidden)
                errors = (prediction - target).square()[mask]
                if len(errors):
                    loss = errors.mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite BC loss")
                    if optimizer is not None:
                        (errors.sum() / max(int(valid.sum()) * 6, 1)).backward()
                    total_squared_error += errors.detach().sum(dim=0).cpu().numpy()
                    total_samples += len(errors)
            if hidden is not None:
                hidden = hidden.detach()
        if optimizer is not None and valid.any():
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    if total_samples == 0:
        raise ValueError("No eligible pre-contact BC samples")
    return {"mse": float(total_squared_error.mean() / total_samples), "axis_rmse": np.sqrt(total_squared_error / total_samples).tolist(), "samples": total_samples}


def main():
    parser = argparse.ArgumentParser(description="Pretrain only E3 base action means on continuous search demonstrations")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial_checkpoint", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.sequence_length, args.learning_rate) <= 0:
        raise ValueError("Epochs, sequence length and learning rate must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    with (args.dataset / "metadata.json").open() as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != 1 or metadata.get("action_indices") != list(range(6)):
        raise ValueError("Unsupported search dataset schema")
    if metadata["collection"].get("bc_checkpoint"):
        raise ValueError("Policy evaluation trajectories are not expert BC labels")
    files = sorted(args.dataset.glob("trajectory_*.npz"))
    if len(files) < 2:
        raise ValueError("At least two trajectories are required for a held-out validation trajectory")
    args.output.mkdir(parents=True, exist_ok=False)
    env_cfg, train_cfg = metadata["env"], metadata["train"]
    model = ActorCriticMultimodal(
        (sum(env_cfg["obs_dim"].values()),), (24,),
        train_cfg["learn"]["init_noise_std"], train_cfg["policy"], train_cfg["encoder"], env_cfg,
    ).to(args.device)
    if args.initial_checkpoint:
        model.load_state_dict(torch.load(args.initial_checkpoint, map_location=args.device))
    for name, parameter in model.named_parameters():
        if name.startswith(("critic.", "log_std_head.")) or name == "log_std":
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    training_files, validation_files = files[:-1], files[-1:]
    metrics = []
    best_loss = float("inf")
    initial = trajectory_pass(model, validation_files, args.device, args.sequence_length)
    print("Initial validation:", json.dumps(initial), flush=True)
    for epoch in range(args.epochs):
        generator.shuffle(training_files)
        training = trajectory_pass(model, training_files, args.device, args.sequence_length, optimizer)
        validation = trajectory_pass(model, validation_files, args.device, args.sequence_length)
        entry = {"epoch": epoch + 1, "training": training, "validation": validation}
        metrics.append(entry)
        print(json.dumps(entry), flush=True)
        if validation["mse"] < best_loss:
            best_loss = validation["mse"]
            torch.save(model.state_dict(), args.output / "model_0.pt")
        with (args.output / "metrics.json").open("w") as handle:
            json.dump({"initial_validation": initial, "epochs": metrics}, handle, indent=2)
    with (args.output / "pretraining.json").open("w") as handle:
        json.dump({"args": vars(args), "env": env_cfg, "train": train_cfg, "training_files": training_files, "validation_files": validation_files, "loss_action_indices": list(range(6))}, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
