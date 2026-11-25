"""Separate fixed-round counting uncertainty from observed round dispersion.

The bootstrap array has one row per observed round. Averaging a draw from each
row conditions on those rounds; it does not resample already-noisy round means.
Variance subtraction is a descriptive random-effects diagnostic, not a confidence
interval or an estimate of stationarity from three jobs.
"""
from __future__ import annotations

import numpy as np


def conditional_mean_draws(draws: np.ndarray, log_scale: bool = False) -> np.ndarray:
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2 or min(draws.shape) < 2 or not np.isfinite(draws).all():
        raise ValueError("Expected finite round-by-bootstrap draws.")
    if log_scale:
        if np.any(draws <= 0):
            raise ValueError("Log-scale bootstrap draws must be positive.")
        return np.exp(np.log(draws).mean(axis=0))
    return draws.mean(axis=0)


def variance_components(values: np.ndarray, shot_variances: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    shot_variances = np.asarray(shot_variances, dtype=float)
    if values.ndim != 1 or len(values) < 2 or values.shape != shot_variances.shape:
        raise ValueError("Each round needs a point estimate and a shot variance.")
    if not np.isfinite(values).all() or not np.isfinite(shot_variances).all():
        raise ValueError("Non-finite variance input.")
    if np.any(shot_variances < 0):
        raise ValueError("Negative shot variance.")
    observed = float(values.var(ddof=1))
    shot = float(shot_variances.mean())
    excess = max(0.0, observed - shot)
    return {
        "observed_round_variance": observed,
        "mean_conditional_shot_variance": shot,
        "excess_round_variance_method_of_moments": excess,
        "conditional_shot_variance_of_mean": shot / len(values),
        "working_variance_of_mean_method_of_moments": (shot + excess) / len(values),
    }
