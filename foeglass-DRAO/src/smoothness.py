"""
src/smoothness.py
=================
Temporal smoothness penalty over frame-level embeddings.

Mathematical definition
-----------------------
For a sequence of frame embeddings z = (z_0, z_1, …, z_{T−1}) and a
set of lags L = {1, 5} (short- and medium-range coherence):

  S(z) = Σ_{l ∈ L}  Σ_{t=l}^{T−1}  ‖z_t − z_{t−l}‖²

This is a multi-scale energy measure of temporal irregularity.
Naturalness corresponds to *lower* S, so S enters the DRAO objective
with a negative sign:

  J_t = Δf_t − λ1·ΔD_t − λ2·S_t

Lag 1 captures frame-to-frame jitter.
Lag 5 captures syllable-scale smoothness.
"""
from __future__ import annotations

import torch


def _lagged_energy(
    frames: torch.Tensor,
    lag: int,
) -> torch.Tensor:
    """
    Compute  Σ_{t=lag}^{T−1}  ‖z_t − z_{t−lag}‖²

    If the sequence is shorter than or equal to `lag`, returns 0.

    Parameters
    ----------
    frames : (T, D)
    lag    : positive integer

    Returns
    -------
    Scalar tensor.
    """
    if frames.shape[0] <= lag:
        return torch.tensor(0.0, device=frames.device, dtype=frames.dtype)
    diff = frames[lag:] - frames[:-lag]          # (T−lag, D)
    return (diff ** 2).sum(dim=1).sum()          # scalar


def multiscale_temporal_smoothness(
    frame_embeddings: torch.Tensor,
    lags: tuple[int, ...] = (1, 5),
) -> torch.Tensor:
    """
    Multi-scale temporal smoothness penalty.

    S(z) = Σ_{l ∈ lags}  Σ_{t=l}^{T−1}  ‖z_t − z_{t−l}‖²

    Parameters
    ----------
    frame_embeddings : Tensor of shape (T, D)
        Per-frame hidden states from wav2vec 2.0.
    lags : tuple of ints
        Lag values in frames.  Default: (1, 5).

    Returns
    -------
    Scalar tensor  ≥ 0.

    Raises
    ------
    ValueError : if frame_embeddings is not 2-D.
    """
    if frame_embeddings.ndim != 2:
        raise ValueError(
            f"frame_embeddings must be 2-D (T, D); got shape {tuple(frame_embeddings.shape)}"
        )
    T = frame_embeddings.shape[0]
    if T < 2:
        # Single frame — smoothness is trivially zero
        return torch.tensor(0.0, device=frame_embeddings.device, dtype=frame_embeddings.dtype)

    total_energy = sum(_lagged_energy(frame_embeddings, lag) for lag in lags)
    total_pairs = sum(max(T - lag, 0) for lag in lags)
    if total_pairs <= 0:
        return torch.tensor(0.0, device=frame_embeddings.device, dtype=frame_embeddings.dtype)
    return total_energy / total_pairs
