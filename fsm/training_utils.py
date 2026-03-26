"""
Training-time utilities for Fisher Score Matching.

Provides:
- set_seed        : reproducible seeding across Python / NumPy / PyTorch / JAX
- check_finite    : NaN / Inf fail-fast guard
- clip_gradients  : optional gradient-norm clipping helper
"""

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int, seed_jax: bool = True) -> None:
    """Set random seeds for Python, NumPy, PyTorch (and optionally JAX).

    Parameters
    ----------
    seed:
        Integer seed.  Must be in ``[0, 2**31 - 1]``.
    seed_jax:
        If ``True`` and JAX is importable, seed JAX via
        ``jax.random.PRNGKey(seed)`` and store it in the module-level variable
        ``fsm.training_utils.JAX_KEY`` for later use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if seed_jax:
        try:
            import jax
            global JAX_KEY
            JAX_KEY = jax.random.PRNGKey(seed)
        except ImportError:
            pass


JAX_KEY = None  # populated by set_seed(seed, jax_seed=True)


# ---------------------------------------------------------------------------
# NaN / Inf guards
# ---------------------------------------------------------------------------

def check_finite(
    tensor: torch.Tensor,
    name: str = "tensor",
    raise_on_error: bool = True,
) -> bool:
    """Check that ``tensor`` contains no NaN or Inf values.

    Parameters
    ----------
    tensor:
        PyTorch tensor to inspect.
    name:
        Human-readable label used in error / warning messages.
    raise_on_error:
        If ``True`` (default) raise :class:`RuntimeError` on failure.
        If ``False`` return ``False`` instead.

    Returns
    -------
    bool
        ``True`` if all values are finite, ``False`` otherwise (only when
        ``raise_on_error=False``).
    """
    if not torch.isfinite(tensor).all():
        msg = f"Non-finite values detected in '{name}' (NaN or Inf)."
        if raise_on_error:
            raise RuntimeError(msg)
        return False
    return True


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def clip_gradients(
    model: nn.Module,
    max_norm: float,
) -> float:
    """Clip gradients of ``model`` parameters by global L2 norm.

    Call this *after* ``loss.backward()`` and *before* ``optimizer.step()``.

    Parameters
    ----------
    model:
        The neural network whose parameter gradients to clip.
    max_norm:
        Maximum allowed gradient norm.  Typically ``1.0`` or ``5.0``.

    Returns
    -------
    float
        Total gradient norm *before* clipping.
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()
