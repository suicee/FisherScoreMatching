"""
Tests for fsm.divergence (PyTorch) and fsm.divergence_jax (JAX).

Covers:
- Shape correctness of outputs
- Finiteness of outputs
- Analytic ground truth: f(x) = A x  => div(f)(x) = trace(A)
"""

import pytest
import numpy as np
import torch

# ---------------------------------------------------------------------------
# PyTorch tests
# ---------------------------------------------------------------------------

from fsm.divergence import hutchinson_divergence as torch_hutchinson_div
from fsm.divergence import exact_divergence as torch_exact_div


def _make_linear_fn_torch(A: torch.Tensor):
    """Return f(x) = A @ x as a callable (batch-aware)."""
    def fn(x):
        # x: (B, D), A: (D, D) -> output: (B, D)
        return x @ A.T
    return fn


class TestTorchDivergence:

    def test_output_shape(self):
        """hutchinson_divergence returns shape (B,)."""
        B, D = 8, 5
        x = torch.randn(B, D, requires_grad=True)
        fn = _make_linear_fn_torch(torch.eye(D))
        div = torch_hutchinson_div(fn, x)
        assert div.shape == (B,), f"expected ({B},), got {div.shape}"

    def test_output_finite(self):
        """All outputs are finite (no NaN / Inf)."""
        B, D = 16, 4
        x = torch.randn(B, D, requires_grad=True)
        fn = _make_linear_fn_torch(torch.randn(D, D))
        div = torch_hutchinson_div(fn, x, n_probes=4, noise="rademacher")
        assert torch.isfinite(div).all(), "Non-finite values in divergence estimate"

    def test_gaussian_noise_shape_and_finite(self):
        """Gaussian probe variant also produces correct shape and finite output."""
        B, D = 8, 3
        x = torch.randn(B, D, requires_grad=True)
        fn = _make_linear_fn_torch(torch.eye(D))
        div = torch_hutchinson_div(fn, x, n_probes=2, noise="gaussian")
        assert div.shape == (B,)
        assert torch.isfinite(div).all()

    def test_analytic_identity(self):
        """For f(x) = I x, divergence should equal D everywhere."""
        B, D = 32, 6
        A = torch.eye(D)
        expected_div = float(D)  # trace(I) = D
        x = torch.randn(B, D, requires_grad=True)
        fn = _make_linear_fn_torch(A)

        # Exact divergence
        exact = torch_exact_div(fn, x)
        assert exact.shape == (B,)
        np.testing.assert_allclose(
            exact.detach().numpy(), expected_div, atol=1e-4,
            err_msg="exact_divergence: trace(I) should equal D"
        )

        # Hutchinson (many probes => low variance)
        g = torch.Generator()
        g.manual_seed(42)
        hutch = torch_hutchinson_div(fn, x, n_probes=256, noise="rademacher", generator=g)
        np.testing.assert_allclose(
            hutch.detach().numpy(), expected_div, atol=0.5,
            err_msg="hutchinson_divergence: mean estimate should be close to D"
        )

    def test_analytic_general_A(self):
        """For f(x) = A x, exact divergence = trace(A) for all x."""
        B, D = 16, 4
        rng = np.random.default_rng(0)
        A_np = rng.standard_normal((D, D))
        A = torch.tensor(A_np, dtype=torch.float32)
        expected_div = float(np.trace(A_np))

        x = torch.randn(B, D, requires_grad=True)
        fn = _make_linear_fn_torch(A)

        exact = torch_exact_div(fn, x)
        np.testing.assert_allclose(
            exact.detach().numpy(), expected_div, atol=1e-4,
            err_msg="exact_divergence: should equal trace(A)"
        )

    def test_invalid_noise_raises(self):
        """Passing an invalid noise type raises ValueError."""
        x = torch.randn(4, 3, requires_grad=True)
        fn = _make_linear_fn_torch(torch.eye(3))
        with pytest.raises(ValueError, match="noise must be"):
            torch_hutchinson_div(fn, x, noise="invalid")

    def test_exact_div_shape(self):
        """exact_divergence returns shape (B,)."""
        B, D = 5, 7
        x = torch.randn(B, D, requires_grad=True)
        fn = _make_linear_fn_torch(torch.eye(D))
        div = torch_exact_div(fn, x)
        assert div.shape == (B,)


# ---------------------------------------------------------------------------
# JAX tests
# ---------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp
    from fsm.divergence_jax import hutchinson_divergence as jax_hutchinson_div
    from fsm.divergence_jax import exact_divergence as jax_exact_div

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


@pytest.mark.skipif(not _JAX_AVAILABLE, reason="JAX not installed")
class TestJaxDivergence:

    def _linear_fn(self, A):
        """Return f(x) = A @ x as a callable on single x of shape (D,)."""
        def fn(x):
            return A @ x
        return fn

    def test_output_shape(self):
        """hutchinson_divergence returns shape (B,)."""
        B, D = 8, 5
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (B, D))
        A = jnp.eye(D)
        fn = self._linear_fn(A)
        div = jax_hutchinson_div(fn, x, key=jax.random.PRNGKey(1))
        assert div.shape == (B,), f"expected ({B},), got {div.shape}"

    def test_output_finite(self):
        """All outputs are finite (no NaN / Inf)."""
        B, D = 16, 4
        key = jax.random.PRNGKey(7)
        x = jax.random.normal(key, (B, D))
        A = jax.random.normal(jax.random.PRNGKey(1), (D, D))
        fn = self._linear_fn(A)
        div = jax_hutchinson_div(fn, x, n_probes=4, key=jax.random.PRNGKey(2))
        assert jnp.isfinite(div).all(), "Non-finite values in JAX divergence estimate"

    def test_gaussian_noise_shape_and_finite(self):
        """Gaussian probe variant produces correct shape and finite output."""
        B, D = 8, 3
        key = jax.random.PRNGKey(99)
        x = jax.random.normal(key, (B, D))
        A = jnp.eye(D)
        fn = self._linear_fn(A)
        div = jax_hutchinson_div(fn, x, n_probes=2, noise="gaussian", key=jax.random.PRNGKey(3))
        assert div.shape == (B,)
        assert jnp.isfinite(div).all()

    def test_analytic_identity(self):
        """For f(x) = I x, divergence should equal D everywhere."""
        B, D = 32, 6
        A = jnp.eye(D)
        expected_div = float(D)
        key = jax.random.PRNGKey(5)
        x = jax.random.normal(key, (B, D))
        fn = self._linear_fn(A)

        # Exact
        exact = jax_exact_div(fn, x)
        assert exact.shape == (B,)
        np.testing.assert_allclose(
            np.array(exact), expected_div, atol=1e-4,
            err_msg="jax exact_divergence: trace(I) should equal D"
        )

        # Hutchinson (many probes)
        hutch = jax_hutchinson_div(fn, x, n_probes=256, key=jax.random.PRNGKey(6))
        np.testing.assert_allclose(
            np.array(hutch), expected_div, atol=0.5,
            err_msg="jax hutchinson_divergence: mean estimate should be close to D"
        )

    def test_analytic_general_A(self):
        """For f(x) = A x, exact divergence = trace(A) for all x."""
        B, D = 16, 4
        rng = np.random.default_rng(0)
        A_np = rng.standard_normal((D, D)).astype(np.float32)
        A = jnp.array(A_np)
        expected_div = float(np.trace(A_np))

        key = jax.random.PRNGKey(10)
        x = jax.random.normal(key, (B, D))
        fn = self._linear_fn(A)

        exact = jax_exact_div(fn, x)
        np.testing.assert_allclose(
            np.array(exact), expected_div, atol=1e-4,
            err_msg="jax exact_divergence: should equal trace(A)"
        )

    def test_invalid_noise_raises(self):
        """Passing an invalid noise type raises ValueError."""
        x = jnp.ones((4, 3))
        fn = self._linear_fn(jnp.eye(3))
        with pytest.raises(ValueError, match="noise must be"):
            jax_hutchinson_div(fn, x, noise="bad")

    def test_exact_div_shape(self):
        """exact_divergence returns shape (B,)."""
        B, D = 5, 7
        key = jax.random.PRNGKey(20)
        x = jax.random.normal(key, (B, D))
        fn = self._linear_fn(jnp.eye(D))
        div = jax_exact_div(fn, x)
        assert div.shape == (B,)


# ---------------------------------------------------------------------------
# Training utilities tests
# ---------------------------------------------------------------------------

from fsm.training_utils import set_seed, check_finite, clip_gradients


class TestTrainingUtils:

    def test_set_seed_reproducibility(self):
        """set_seed produces reproducible torch random samples."""
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b), "set_seed must produce reproducible results"

    def test_check_finite_passes(self):
        """check_finite returns True for finite tensors."""
        t = torch.tensor([1.0, 2.0, 3.0])
        assert check_finite(t, "test") is True

    def test_check_finite_raises_nan(self):
        """check_finite raises RuntimeError on NaN."""
        t = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(RuntimeError, match="Non-finite"):
            check_finite(t, "bad_tensor")

    def test_check_finite_raises_inf(self):
        """check_finite raises RuntimeError on Inf."""
        t = torch.tensor([1.0, float("inf")])
        with pytest.raises(RuntimeError):
            check_finite(t)

    def test_check_finite_no_raise(self):
        """check_finite returns False without raising when raise_on_error=False."""
        t = torch.tensor([float("nan")])
        result = check_finite(t, raise_on_error=False)
        assert result is False

    def test_clip_gradients(self):
        """clip_gradients reduces gradient norm to at most max_norm."""
        import torch.nn as nn
        model = nn.Linear(10, 10)
        x = torch.randn(5, 10)
        loss = model(x).sum()
        loss.backward()

        max_norm = 0.1
        clip_gradients(model, max_norm)

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        assert total_norm <= max_norm + 1e-6, (
            f"Gradient norm {total_norm:.4f} exceeds max_norm {max_norm}"
        )
