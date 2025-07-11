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

We emphasize that these experiments are primarily designed to evaluate the feasibility of the method. There are still many practical aspects we are actively exploring. Our goal is to develop a robust method and package that can be reliably applied to a wide range of models.

Feel free to contact me at suic20@mails.tsinghua.edu.cn or csui1@jhu.edu if you’d like assistance applying this approach to other examples. You’re also welcome to open an issue or pull request if you encounter any problems or have suggestions!