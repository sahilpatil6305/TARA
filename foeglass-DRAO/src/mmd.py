"""
src/mmd.py
==========
Maximum Mean Discrepancy (MMD) with an adaptive RBF kernel.

Mathematical definition
-----------------------
Given two sets of embeddings X = {z_i} and Y = {r_j}:

  MMD²(X, Y) = E[k(z_i, z_j)]
             + E[k(r_i, r_j)]
             - 2·E[k(z_i, r_j)]

where the RBF (Gaussian) kernel is:

  k(a, b) = exp( -‖a − b‖² / σ² )

Bandwidth σ is selected via the **median heuristic**:

  σ = sqrt( median { ‖x_i − x_j‖² : i < j, x ∈ X ∪ Y } )

This choice makes MMD invariant to the absolute scale of embeddings
and is standard in the kernel two-sample test literature.

Role in DRAO
------------
D_t = MMD²( Z_gen_buffer_t, Z_real_subset )

ΔD_t = D_t − D_{t−1}  is penalised in the objective:
  J_t = Δf_t − λ1·ΔD_t − λ2·S_t
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Kernel internals
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise_sq_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute ‖x_i − y_j‖² for all (i, j) without forming explicit differences.

    Uses the identity:  ‖a − b‖² = ‖a‖² + ‖b‖² − 2⟨a, b⟩

    Parameters
    ----------
    x : (M, D)
    y : (N, D)

    Returns
    -------
    dist : (M, N)  — clipped at 0 to prevent numerical negatives.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xx = np.sum(x * x, axis=1, keepdims=True)          # (M, 1)
    yy = np.sum(y * y, axis=1, keepdims=True).T        # (1, N)
    dist = xx + yy - 2.0 * (x @ y.T)
    return np.clip(dist, 0.0, None)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with epsilon protection."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D array for row normalization; got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def _median_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    """
    Median heuristic for RBF bandwidth.

    Returns σ = sqrt( median(upper-triangle pairwise sq-dists in X∪Y) + ε )
    Falls back to 1.0 when the combined set is a singleton.
    """
    combined = np.concatenate([x, y], axis=0)
    dists = _pairwise_sq_dist(combined, combined)
    idx = np.triu_indices(dists.shape[0], k=1)
    upper = dists[idx]
    upper = upper[upper > 0.0]
    if upper.size == 0:
        return 1.0
    return float(np.sqrt(np.median(upper) + 1e-12))


def _rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """
    k(x_i, y_j) = exp( −‖x_i − y_j‖² / σ² )

    Returns
    -------
    K : (M, N)
    """
    return np.exp(-_pairwise_sq_dist(x, y) / (sigma ** 2 + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def batch_mmd(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute the unbiased MMD² estimator between two batches of embeddings.

    The unbiased estimator excludes the diagonal self-similarity terms from
    Kxx and Kyy:

      MMD²_u(X, Y) =
          (1 / (m (m - 1))) Σ_{i != j} k(x_i, x_j)
        + (1 / (n (n - 1))) Σ_{i != j} k(y_i, y_j)
        - (2 / (m n)) Σ_{i, j} k(x_i, y_j)

    Parameters
    ----------
    x : (M, D)  generated embeddings
    y : (N, D)  real embeddings

    Returns
    -------
    (mmd_sq, bandwidth)
        mmd_sq    — MMD²_u(x, y), clipped at 0 for numerical stability
        bandwidth — σ used for the kernel (useful for diagnostics)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            f"batch_mmd expects 2-D arrays [batch, dim]; "
            f"got x={x.shape}, y={y.shape}"
        )
    if len(x) < 2 or len(y) < 2:
        raise ValueError(
            "Unbiased MMD requires at least 2 samples in each batch; "
            f"got len(x)={len(x)}, len(y)={len(y)}"
        )

    x = _l2_normalize_rows(x)
    y = _l2_normalize_rows(y)

    sigma = _median_bandwidth(x, y)
    k_xx = _rbf_kernel(x, x, sigma)
    k_yy = _rbf_kernel(y, y, sigma)
    k_xy = _rbf_kernel(x, y, sigma)

    m = len(x)
    n = len(y)
    mean_xx = float((k_xx.sum() - np.trace(k_xx)) / (m * (m - 1)))
    mean_yy = float((k_yy.sum() - np.trace(k_yy)) / (n * (n - 1)))
    mean_xy = float(k_xy.mean())
    mmd_sq = mean_xx + mean_yy - 2.0 * mean_xy
    return float(max(mmd_sq, 0.0)), sigma


# ─────────────────────────────────────────────────────────────────────────────
# Rolling buffer for generated embeddings
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingBuffer:
    """
    Fixed-capacity FIFO buffer that accumulates generated global embeddings.

    Used to estimate D_t = MMD²(Z_gen_buffer, Z_real_subset) at each
    pipeline iteration.  Older embeddings fall off as new ones arrive.

    Attributes
    ----------
    max_size : int
        Maximum number of embeddings retained (rolling window).
    """
    max_size: int
    _values: Deque[np.ndarray] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        self._values = deque(maxlen=self.max_size)

    def append(self, embedding: np.ndarray) -> None:
        """Add one global embedding (shape (D,)) to the buffer."""
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(emb)
        if norm > 0.0:
            emb = emb / norm
        self._values.append(emb)

    def as_array(self) -> np.ndarray:
        """Stack contents into (N, D).  Raises if buffer is empty."""
        if not self._values:
            raise ValueError("EmbeddingBuffer is empty — cannot form array.")
        return np.stack(list(self._values), axis=0)

    def __len__(self) -> int:
        return len(self._values)

    def is_ready(self, min_size: int = 2) -> bool:
        """Return True when enough embeddings have accumulated."""
        return len(self._values) >= min_size
