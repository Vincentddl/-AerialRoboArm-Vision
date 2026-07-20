from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a short-horizon bearing GRU estimator.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR
        / "datasets"
        / "bearing_estimation"
        / "v1"
        / "bearing_sequences.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "models" / "bearing_estimator_v1.pt",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_DIR / "outputs" / "bearing_estimator_v1_training.json",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    return parser.parse_args()


class BearingDataset(Dataset):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        indices: np.ndarray,
        normalization: dict,
    ):
        self.features = arrays["features"][indices].copy()
        self.lengths = arrays["lengths"][indices].copy()
        self.horizons = arrays["horizons"][indices].copy()
        self.targets = arrays["targets"][indices].copy()
        self.weights = arrays["weights"][indices].copy()
        self.feature_mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(normalization["feature_std"], dtype=np.float32)
        self.horizon_mean = float(normalization["horizon_mean"])
        self.horizon_std = float(normalization["horizon_std"])
        self.target_mean = np.asarray(normalization["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(normalization["target_std"], dtype=np.float32)

        for index, length in enumerate(self.lengths):
            self.features[index, :length] = (
                self.features[index, :length] - self.feature_mean
            ) / self.feature_std
            self.features[index, length:] = 0.0
        self.horizons = (self.horizons - self.horizon_mean) / self.horizon_std
        self.targets = (self.targets - self.target_mean) / self.target_std

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.features[index]),
            torch.tensor(self.lengths[index], dtype=torch.long),
            torch.from_numpy(self.horizons[index]),
            torch.from_numpy(self.targets[index]),
            torch.tensor(self.weights[index], dtype=torch.float32),
        )


class BearingGRU(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_size, 2),
        )

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor, horizons: torch.Tensor
    ) -> torch.Tensor:
        packed = pack_padded_sequence(
            features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.head(torch.cat((hidden[-1], horizons), dim=1))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_arrays(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def compute_normalization(arrays: Dict[str, np.ndarray], train_indices: np.ndarray) -> dict:
    valid_features = []
    for index in train_indices:
        valid_features.append(arrays["features"][index, : arrays["lengths"][index]])
    feature_values = np.concatenate(valid_features, axis=0)
    targets = arrays["targets"][train_indices]
    horizons = arrays["horizons"][train_indices]
    return {
        "feature_mean": feature_values.mean(axis=0).tolist(),
        "feature_std": np.maximum(feature_values.std(axis=0), 1e-4).tolist(),
        "horizon_mean": float(horizons.mean()),
        "horizon_std": max(float(horizons.std()), 1e-4),
        "target_mean": targets.mean(axis=0).tolist(),
        "target_std": np.maximum(targets.std(axis=0), 1e-4).tolist(),
    }


def weighted_loss(
    predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    per_value = nn.functional.smooth_l1_loss(predictions, targets, reduction="none")
    per_sample = per_value.mean(dim=1)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def run_epoch(
    model: BearingGRU,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    for features, lengths, horizons, targets, weights in loader:
        features = features.to(device)
        lengths = lengths.to(device)
        horizons = horizons.to(device)
        targets = targets.to(device)
        weights = weights.to(device)
        with torch.set_grad_enabled(training):
            predictions = model(features, lengths, horizons)
            loss = weighted_loss(predictions, targets, weights)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.detach()) * len(features)
        total_items += len(features)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def predict_dataset(
    model: BearingGRU,
    loader: DataLoader,
    device: torch.device,
    normalization: dict,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    target_mean = np.asarray(normalization["target_mean"], dtype=np.float32)
    target_std = np.asarray(normalization["target_std"], dtype=np.float32)
    for features, lengths, horizons, batch_targets, _ in loader:
        output = model(features.to(device), lengths.to(device), horizons.to(device))
        predictions.append(output.cpu().numpy() * target_std + target_mean)
        targets.append(batch_targets.numpy() * target_std + target_mean)
    return np.concatenate(predictions), np.concatenate(targets)


def error_summary(values: np.ndarray) -> dict:
    return {
        "count": int(len(values)),
        "median_deg": round(float(np.median(values)), 4),
        "p90_deg": round(float(np.percentile(values, 90)), 4),
        "mean_deg": round(float(np.mean(values)), 4),
    }


def evaluate_split(
    model: BearingGRU,
    loader: DataLoader,
    device: torch.device,
    normalization: dict,
    raw_horizons: np.ndarray,
) -> dict:
    predictions, targets = predict_dataset(model, loader, device, normalization)
    estimator_error = np.linalg.norm(predictions - targets, axis=1)
    hold_error = np.linalg.norm(targets, axis=1)
    result = {
        "learned_estimator": error_summary(estimator_error),
        "hold_current_angle": error_summary(hold_error),
        "median_improvement_vs_hold_percent": round(
            100.0 * (float(np.median(hold_error)) - float(np.median(estimator_error)))
            / max(float(np.median(hold_error)), 1e-6),
            2,
        ),
        "by_horizon_ms": {},
    }
    flat_horizons = np.rint(raw_horizons.reshape(-1) * 1000).astype(int)
    for horizon_ms in sorted(set(flat_horizons.tolist())):
        mask = flat_horizons == horizon_ms
        result["by_horizon_ms"][str(horizon_ms)] = {
            "learned_estimator": error_summary(estimator_error[mask]),
            "hold_current_angle": error_summary(hold_error[mask]),
        }
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    arrays = load_arrays(args.data)
    indices = {
        "train": np.flatnonzero(arrays["splits"] == 0),
        "validation": np.flatnonzero(arrays["splits"] == 1),
        "test": np.flatnonzero(arrays["splits"] == 2),
    }
    if any(len(value) == 0 for value in indices.values()):
        raise ValueError(f"every grouped split must contain samples: {indices}")

    normalization = compute_normalization(arrays, indices["train"])
    datasets = {
        name: BearingDataset(arrays, split_indices, normalization)
        for name, split_indices in indices.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=args.batch_size, shuffle=False
        ),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
    }

    model = BearingGRU(input_size=arrays["features"].shape[2], hidden_size=args.hidden_size).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = None
    best_validation = math.inf
    stale_epochs = 0
    history = []

    print(
        f"device={device} train={len(indices['train'])} "
        f"validation={len(indices['validation'])} test={len(indices['test'])}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], device, optimizer)
        validation_loss = run_epoch(model, loaders["validation"], device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:03d} train={train_loss:.5f} "
                f"validation={validation_loss:.5f}"
            )
        if stale_epochs >= args.patience:
            print(f"early stop at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    metrics = {}
    for name in ("validation", "test"):
        metrics[name] = evaluate_split(
            model,
            loaders[name],
            device,
            normalization,
            arrays["horizons"][indices[name]],
        )

    checkpoint = {
        "state_dict": best_state,
        "model": {
            "type": "BearingGRU",
            "input_size": int(arrays["features"].shape[2]),
            "hidden_size": args.hidden_size,
            "output": "future bearing delta yaw/pitch in degrees",
        },
        "normalization": normalization,
        "horizons_ms": sorted(
            set(np.rint(arrays["horizons"].reshape(-1) * 1000).astype(int).tolist())
        ),
        "source_dataset": str(args.data.resolve()),
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "device": str(device),
        "torch_version": torch.__version__,
        "samples": {name: int(len(value)) for name, value in indices.items()},
        "best_validation_loss": best_validation,
        "epochs_completed": len(history),
        "metrics": metrics,
        "training_history": history,
        "checkpoint": str(args.output.resolve()),
        "warning": "Metrics use V7 pseudo-label centers and are not mechanical ground truth.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"checkpoint: {args.output.resolve()}")
    print(f"report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
