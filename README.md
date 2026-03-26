# Fisher Score Matching

This repository contains experimental code for the project **Fisher Score Matching for Simulation-Based Forecasting and Inference**. In this work, we study how to apply score matching to learn the Fisher score, defined as

$$
s(x, \theta) = \nabla_{\theta} \log P(x \mid \theta)
$$

This quantity is useful for:

- Maximum likelihood estimation  
- Learning locally sufficient summary statistics  
- Fisher forecasting  
- Simulation-based inference (SBI)

For theoretical details, please refer to our [paper](https://arxiv.org/abs/2507.07833).

## Code Structure

The core modules are in the `fsm` folder. We include three example notebooks demonstrating how to apply the method to different tasks:

1. **`simple_gaussian.ipynb`**  
   Demonstrates how to apply Fisher Score Matching to a decomposable Gaussian model and estimate the Fisher score.

2. **`two_moon.ipynb`**  
   Applies the method to the standard SBI benchmark task *two-moon*, showing that it can be used for performing SBI on models that are not decomposable.

3. **`jaxcosmo_Cl.ipynb`**  
   Applies the method to a weak lensing inference problem using `jax-cosmo`. We compare our approach to a differentiable simulator and show how it can be used for Fisher forecasting and Bayesian inference.

---

## Utilities

### Divergence Estimator (Hutchinson Trace Estimator)

Computing the divergence of a vector field `f: R^D → R^D` naively requires forming the full `D×D` Jacobian, which is `O(D²)` in memory. We provide a memory-efficient Hutchinson trace estimator that only needs a single VJP (PyTorch) or JVP (JAX) per probe vector:

```
div(f)(x) = trace(J_f(x)) ≈ E_v[ v^T J_f(x) v ]
```

where `v` is a random probe vector with `E[vvᵀ] = I` (Rademacher ±1 or Gaussian).

#### PyTorch

```python
import torch
from fsm.divergence import hutchinson_divergence, exact_divergence

# Define a vector field f: (B, D) -> (B, D)
A = torch.randn(4, 4)
def f(x):
    return x @ A.T

x = torch.randn(32, 4, requires_grad=True)

# Hutchinson estimate (fast, stochastic) — shape (B,)
div_est = hutchinson_divergence(f, x, n_probes=4, noise="rademacher")

# Exact divergence via diagonal Jacobian (slow, deterministic) — shape (B,)
div_exact = exact_divergence(f, x)
```

#### JAX

```python
import jax, jax.numpy as jnp
from fsm.divergence_jax import hutchinson_divergence, exact_divergence

A = jax.random.normal(jax.random.PRNGKey(0), (4, 4))
# fn must operate on a single sample (D,), not a batch
def f(x):
    return A @ x

x = jax.random.normal(jax.random.PRNGKey(1), (32, 4))
key = jax.random.PRNGKey(2)

# Hutchinson estimate — shape (B,)
div_est = hutchinson_divergence(f, x, n_probes=4, key=key)

# Exact divergence — shape (B,)
div_exact = exact_divergence(f, x)
```

### Training Utilities

```python
from fsm.training_utils import set_seed, check_finite, clip_gradients

# Reproducible seeding (Python / NumPy / PyTorch / JAX)
set_seed(42)

# Fail-fast NaN/Inf guard — raises RuntimeError on bad values
check_finite(loss, name="loss")

# Gradient norm clipping (call after loss.backward(), before optimizer.step())
grad_norm = clip_gradients(model, max_norm=1.0)
```

The `FSM_Regressor.train()` method now accepts `max_grad_norm` and `check_nan` arguments
to enable these guards during training:

```python
fm_regressor.train(
    train_input, train_output,
    lr=1e-3, epochs=1000,
    max_grad_norm=1.0,   # clip gradients
    check_nan=True,      # raise on NaN loss
    verbose=True,
)
```

---

## Running Tests

```bash
pip install pytest torch jax[cpu]
pytest tests/ -v
```

---

We emphasize that these experiments are primarily designed to evaluate the feasibility of the method. There are still many practical aspects we are actively exploring. Our goal is to develop a robust method and package that can be reliably applied to a wide range of models.

Feel free to contact me at suic20@mails.tsinghua.edu.cn or csui1@jhu.edu if you’d like assistance applying this approach to other examples. You’re also welcome to open an issue or pull request if you encounter any problems or have suggestions!