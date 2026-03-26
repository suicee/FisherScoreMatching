"""
PyTorch Hutchinson trace estimator for divergence of a vector field.

The divergence of a vector field f: R^D -> R^D is:
    div(f)(x) = trace(J_f(x))   where J_f is the Jacobian of f.

Computing the full D×D Jacobian is O(D^2) in memory and O(D) forward passes.
The Hutchinson estimator approximates the trace cheaply:
    trace(J) ≈ E_v[ v^T J v ]   with E[v v^T] = I   (e.g. Rademacher or Gaussian v).

We exploit the VJP (reverse-mode) identity:
    v^T J = vjp(f, x)(v)
so a single backward pass gives v^T J, and then dot with v yields v^T J v.

Shapes throughout:
    x       : (B, D) — batch of input points
    f(x)    : (B, D) — vector field outputs
    v       : (B, D) — Hutchinson probe vectors (one per sample)
    returns : (B,)   — per-sample divergence estimates
"""

import torch


def hutchinson_divergence(
    fn,
    x: torch.Tensor,
    n_probes: int = 1,
    noise: str = "rademacher",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Estimate div(fn)(x) via the Hutchinson trace estimator (VJP-based).

    No full Jacobian is materialised.  A single VJP is computed per probe.

    Parameters
    ----------
    fn:
        Vector field mapping ``(B, D) -> (B, D)``.  Must be differentiable
        w.r.t. ``x``.
    x:
        Input tensor of shape ``(B, D)`` with ``requires_grad=True`` (or the
        function will enable it temporarily).
    n_probes:
        Number of Hutchinson probe vectors to average.  More probes reduce
        variance; ``n_probes=1`` is often sufficient during training.
    noise:
        Probe distribution.  ``"rademacher"`` (±1, lower variance) or
        ``"gaussian"`` (standard normal).
    generator:
        Optional :class:`torch.Generator` for reproducible sampling.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)``.  Per-sample divergence estimates.
    """
    if noise not in ("rademacher", "gaussian"):
        raise ValueError(f"noise must be 'rademacher' or 'gaussian', got '{noise}'")

    x_req = x.requires_grad
    if not x_req:
        x = x.detach().requires_grad_(True)

    estimates = []
    for _ in range(n_probes):
        if noise == "rademacher":
            v = torch.randint(0, 2, x.shape, generator=generator, dtype=x.dtype, device=x.device) * 2 - 1
        else:
            v = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)

        fx = fn(x)  # (B, D)

        # vjp: (v^T J)_i = d/dx_i [fx · v] — one backward per probe
        (vjp,) = torch.autograd.grad(
            fx,
            x,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
        )  # (B, D)

        # v^T J v = sum over D (element-wise product), shape (B,)
        estimates.append((vjp * v).sum(dim=-1))

    return torch.stack(estimates, dim=0).mean(dim=0)  # (B,)


def exact_divergence(fn, x: torch.Tensor) -> torch.Tensor:
    """Compute exact divergence via the full diagonal of the Jacobian.

    Useful for small dimensions and unit testing.  Requires D forward+backward
    passes.

    Parameters
    ----------
    fn:
        Vector field ``(B, D) -> (B, D)``.
    x:
        Shape ``(B, D)``.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)``.  Exact per-sample divergence.
    """
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)

    B, D = x.shape
    diag_sum = torch.zeros(B, dtype=x.dtype, device=x.device)

    for d in range(D):
        fx = fn(x)          # (B, D)
        e_d = torch.zeros_like(fx)
        e_d[:, d] = 1.0     # one-hot output selector

        (grad_d,) = torch.autograd.grad(
            fx,
            x,
            grad_outputs=e_d,
            create_graph=True,
            retain_graph=True,
        )                   # (B, D) — d-th row of Jacobian

        diag_sum = diag_sum + grad_d[:, d]

    return diag_sum         # (B,)
