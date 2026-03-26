"""
JAX Hutchinson trace estimator for divergence of a vector field.

The divergence of f: R^D -> R^D is  div(f)(x) = trace(J_f(x)).

Instead of materialising the full D×D Jacobian we use:
    trace(J) ≈ E_v[ v^T (J v) ]   with E[v v^T] = I.

The JVP (forward-mode) gives  J v = jvp(f, (x,), (v,))[1]  in one pass,
so:  v^T J v  =  dot(v, jvp(f, (x,), (v,))[1]).

We vmap over the batch dimension and optionally over multiple probes.

Shapes throughout:
    x       : (B, D) — batch of input points
    f(x)    : (B, D) — vector field outputs
    v       : (B, D) — Hutchinson probe vectors
    returns : (B,)   — per-sample divergence estimates
"""

from functools import partial

import jax
import jax.numpy as jnp


def _single_probe_div(fn, x_single: jax.Array, v: jax.Array) -> jax.Array:
    """Compute v^T J_f(x) v for a single (x, v) pair using JVP.

    Parameters
    ----------
    fn:
        Vector field ``(D,) -> (D,)`` (un-batched).
    x_single:
        Shape ``(D,)``.
    v:
        Shape ``(D,)``.

    Returns
    -------
    jax.Array
        Scalar — one Hutchinson sample of the divergence.
    """
    _, jvp_val = jax.jvp(fn, (x_single,), (v,))  # jvp_val shape: (D,)
    return jnp.dot(v, jvp_val)  # scalar


def hutchinson_divergence(
    fn,
    x: jax.Array,
    n_probes: int = 1,
    noise: str = "rademacher",
    key: jax.Array | None = None,
) -> jax.Array:
    """Estimate div(fn)(x) via the Hutchinson trace estimator (JVP-based).

    No full Jacobian is materialised.  One JVP per probe per sample.

    Parameters
    ----------
    fn:
        Vector field mapping ``(D,) -> (D,)`` (single sample, un-batched).
        The function is internally vmapped over the batch dimension.
    x:
        Input array of shape ``(B, D)``.
    n_probes:
        Number of Hutchinson probe vectors to average.
    noise:
        Probe distribution: ``"rademacher"`` (±1) or ``"gaussian"``.
    key:
        JAX PRNG key.  If ``None``, a default key is used (not recommended
        for production; always pass a key for reproducibility).

    Returns
    -------
    jax.Array
        Shape ``(B,)``.  Per-sample divergence estimates.
    """
    if noise not in ("rademacher", "gaussian"):
        raise ValueError(f"noise must be 'rademacher' or 'gaussian', got '{noise}'")

    if key is None:
        key = jax.random.PRNGKey(0)

    B, D = x.shape

    # vmap _single_probe_div over batch axis
    batched_probe = jax.vmap(partial(_single_probe_div, fn), in_axes=(0, 0))

    probe_estimates = []
    for i in range(n_probes):
        key, subkey = jax.random.split(key)
        if noise == "rademacher":
            v = jax.random.rademacher(subkey, (B, D), dtype=x.dtype)
        else:
            v = jax.random.normal(subkey, (B, D), dtype=x.dtype)

        probe_estimates.append(batched_probe(x, v))  # (B,)

    return jnp.stack(probe_estimates, axis=0).mean(axis=0)  # (B,)


def exact_divergence(fn, x: jax.Array) -> jax.Array:
    """Compute exact divergence via the full diagonal of the Jacobian.

    Useful for small dimensions and unit testing.

    Parameters
    ----------
    fn:
        Vector field ``(D,) -> (D,)`` (un-batched).
    x:
        Shape ``(B, D)``.

    Returns
    -------
    jax.Array
        Shape ``(B,)``.  Exact per-sample divergence.
    """
    # jacfwd gives J of shape (D_out, D_in) for a single input (D,)
    jac_fn = jax.jacfwd(fn)  # (D,) -> (D, D)
    batched_jac = jax.vmap(jac_fn)   # (B, D) -> (B, D, D)
    J = batched_jac(x)               # (B, D, D)
    return jnp.trace(J, axis1=-2, axis2=-1)  # (B,)
