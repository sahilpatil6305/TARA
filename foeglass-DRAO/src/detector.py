"""
src/detector.py
===============
Binary audio deepfake detector: DetectorMLP.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


LOGGER = logging.getLogger(__name__)


@dataclass
class DetectorTrainingResult:
    model: "DetectorMLP"
    val_targets: np.ndarray
    val_predictions: np.ndarray
    val_probabilities: np.ndarray
    metrics: dict[str, object]
    train_real_count: int
    train_fake_count: int
    balanced_real_count: int
    balanced_fake_count: int
    train_count: int
    eval_count: int
    class_balance_ratio: float
    train_indices: list[int]
    eval_indices: list[int]


class DetectorMLP(nn.Module):
    """Multi-layer perceptron binary classifier."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 128]
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def _balance_embeddings(
    real_embs: np.ndarray,
    gen_embs: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample the larger class so the detector sees a balanced dataset."""

    n_real = len(real_embs)
    n_fake = len(gen_embs)
    target = min(n_real, n_fake)
    rng = np.random.default_rng(seed)

    if n_real > target:
        real_idx = rng.choice(n_real, size=target, replace=False)
        real_embs = real_embs[real_idx]
    if n_fake > target:
        fake_idx = rng.choice(n_fake, size=target, replace=False)
        gen_embs = gen_embs[fake_idx]
    return real_embs, gen_embs


def _make_dataset(
    real_embs: np.ndarray,
    gen_embs: np.ndarray,
    seed: int = 42,
) -> TensorDataset:
    """Combine real and generated embeddings into a shuffled dataset."""

    X = np.concatenate([real_embs, gen_embs], axis=0).astype(np.float32)
    y = np.concatenate(
        [
            np.ones(len(real_embs), dtype=np.float32),
            np.zeros(len(gen_embs), dtype=np.float32),
        ]
    )
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return TensorDataset(
        torch.from_numpy(X[idx]),
        torch.from_numpy(y[idx]).unsqueeze(1),
    )


def _split(
    dataset: TensorDataset,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    """Split a dataset into deterministic train/val subsets."""

    n = len(dataset)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )


def _evaluate_detector(
    model: DetectorMLP,
    dataset: TensorDataset,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Evaluate the detector on a validation dataset."""

    dl_val = DataLoader(dataset, batch_size=batch_size)
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(device)
            batch_probs = model(xb).squeeze(1).cpu().numpy()
            probs.append(batch_probs)
            targets.append(yb.squeeze(1).cpu().numpy())

    val_probabilities = np.concatenate(probs).astype(np.float32)
    val_targets = np.concatenate(targets).astype(np.int64)
    val_predictions = (val_probabilities >= 0.5).astype(np.int64)
    cm = confusion_matrix(val_targets, val_predictions, labels=[0, 1])
    try:
        roc_auc = float(roc_auc_score(val_targets, val_probabilities))
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "accuracy": float(accuracy_score(val_targets, val_predictions)),
        "precision": float(precision_score(val_targets, val_predictions, zero_division=0)),
        "recall": float(recall_score(val_targets, val_predictions, zero_division=0)),
        "f1_score": float(f1_score(val_targets, val_predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
    }
    return val_targets, val_predictions, val_probabilities, metrics


def train_detector(
    real_embs: np.ndarray,
    gen_embs: np.ndarray,
    save_path: Path,
    hidden_dims: List[int] | None = None,
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 7,
    seed: int = 42,
    device: str | None = None,
) -> DetectorTrainingResult:
    """Train a detector and save the best checkpoint."""

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = real_embs.shape[1]
    train_real_count = len(real_embs)
    train_fake_count = len(gen_embs)
    balanced_real_embs, balanced_fake_embs = _balance_embeddings(real_embs, gen_embs, seed)
    balanced_real_count = len(balanced_real_embs)
    balanced_fake_count = len(balanced_fake_embs)
    class_balance_ratio = float(train_real_count / max(train_fake_count, 1))

    LOGGER.info(
        "[detector] Sample counts | real=%d fake=%d | balanced_real=%d balanced_fake=%d",
        train_real_count,
        train_fake_count,
        balanced_real_count,
        balanced_fake_count,
    )

    ds_full = _make_dataset(balanced_real_embs, balanced_fake_embs, seed=seed)
    ds_train, ds_val = _split(ds_full, seed=seed)
    train_indices = list(ds_train.indices)
    eval_indices = list(ds_val.indices)

    train_generator = torch.Generator().manual_seed(seed)
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )

    model = DetectorMLP(input_dim, hidden_dims, dropout).to(dev)
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(ds_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in DataLoader(ds_val, batch_size=batch_size):
                xb, yb = xb.to(dev), yb.to(dev)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= max(len(ds_val), 1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale = 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            stale += 1

        if epoch % 5 == 0 or stale == 0:
            LOGGER.info(
                "epoch %3d/%d | train=%.4f val=%.4f best=%.4f",
                epoch,
                epochs,
                train_loss,
                val_loss,
                best_val_loss,
            )

        if stale >= patience:
            LOGGER.info("Early stop at epoch %d (patience=%d)", epoch, patience)
            break

    model.load_state_dict(torch.load(save_path, map_location=dev))
    LOGGER.info("[detector] Saved best model -> %s", save_path)

    val_targets, val_predictions, val_probabilities, metrics = _evaluate_detector(
        model=model,
        dataset=ds_val,
        batch_size=batch_size,
        device=dev,
    )
    return DetectorTrainingResult(
        model=model,
        val_targets=val_targets,
        val_predictions=val_predictions,
        val_probabilities=val_probabilities,
        metrics=metrics,
        train_real_count=train_real_count,
        train_fake_count=train_fake_count,
        balanced_real_count=balanced_real_count,
        balanced_fake_count=balanced_fake_count,
        train_count=len(ds_train),
        eval_count=len(ds_val),
        class_balance_ratio=class_balance_ratio,
        train_indices=train_indices,
        eval_indices=eval_indices,
    )


def load_detector(
    checkpoint: Path,
    input_dim: int = 768,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.3,
    device: str | None = None,
) -> DetectorMLP:
    """Restore a previously trained detector checkpoint."""

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectorMLP(input_dim, hidden_dims, dropout)
    model.load_state_dict(torch.load(checkpoint, map_location=dev))
    model.to(dev).eval()
    LOGGER.info("[detector] Loaded checkpoint <- %s", checkpoint)
    return model


@torch.inference_mode()
def score(
    model: DetectorMLP,
    embedding: torch.Tensor,
) -> float:
    """Return the detector's real-speech probability for one embedding."""

    model.eval()
    device = next(model.parameters()).device
    x = embedding.float().unsqueeze(0).to(device)
    return float(model(x).squeeze())
