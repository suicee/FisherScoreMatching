import numpy as np
import matplotlib.pyplot as plt
#torch 
import torch
import torch.nn as nn

import torch
import numpy as np
from regressor import to_numpy, to_tensor

import jax
import jax_cosmo as jc
import jax.numpy as jnp

class LinearGaussianModel:
    def __init__(self, cov, dim=None):
        """
        a linear Gaussian model with mean as parameter and fixed covariance.
        cov: (d, d) covariance matrix (shared across all samples)
        """
        if dim is None and isinstance(cov, np.ndarray):
            dim = cov.shape[0]
        elif dim is not None and isinstance(cov, np.ndarray):
            assert cov.shape[0] == dim and cov.shape[1] == dim, "input dimension does not match covariance matrix dimension"
        elif dim is None and isinstance(cov, float):
            raise ValueError("dim must be specified if cov is an float")

        if isinstance(cov, float):
            cov = np.eye(dim)*cov

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cov = to_tensor(cov).to(self.device)
        self.cov_inv = torch.linalg.inv(self.cov)

    def sample(self, cond_x):
        """
        sample from a multivariate Gaussian distribution given a mean vector cond_x and a fixed covariance matrix.
        cond_x: (N, d) array of means
        returns: (N, d) samples from N(cond_x[i], cov)
        """
        cond_x = to_tensor(cond_x).to(self.device)
        dis = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=cond_x, covariance_matrix=self.cov
        )

        x_sample = dis.sample()

        return to_numpy(x_sample)

    def fisher_score(self, x_sample, cond_x):
        """
        compute the Fisher score for given samples and means.
        cond_x: (N, d) array of means
        x_sample: (N, d) array of samples
        returns: (N, d) array of scores
        """
        x_sample = to_tensor(x_sample).to(self.device)
        cond_x = to_tensor(cond_x).to(self.device)
        score = (x_sample - cond_x) @ self.cov_inv.T

        return score.cpu().numpy()
    
    def sample_and_score(self, cond_x):
        """
        sample from the model and compute the Fisher score.
        cond_x: (N, d) array of means
        returns: (N, d) samples from N(cond_x[i], cov) and (N, d) scores
        """
        x_sample = self.sample(cond_x)
        score = self.fisher_score(x_sample, cond_x)

        return x_sample, score

    


from functools import partial
class jax_cosmo_cl_model:
    def __init__(self, scalor=1e7,n_ell=10):
        ''''
        A wrapper for jax-cosmo weak lensing angular power spectrum model.
        scalor: a scalar to normalize the output angular power spectrum.
        n_ell: number of ell values to compute the angular power spectrum.
        
        
        '''

        # Set up the probes and ell values
        nz1 = jc.redshift.smail_nz(1., 2., 1.)
        nz2 = jc.redshift.smail_nz(1., 2., 0.5)
        nzs = [nz1, nz2]
        self.probes = [jc.probes.WeakLensing(nzs, sigma_e=0.26)]

        self.ell = jnp.logspace(1, 3, n_ell)


        # use fixed covariance matrix at fiducial cosmology
        cosmo = jc.Planck15()
        cls = jc.angular_cl.angular_cl(cosmo, self.ell, self.probes)
        mu, cov = jc.angular_cl.gaussian_cl_covariance_and_mean(
            cosmo, self.ell, self.probes, sparse=True)

        self.scalor = scalor
        self.cov = cov*self.scalor**2
        self.cov_inv = jc.sparse.inv(self.cov)

        self.cov = jc.sparse.to_dense(self.cov)
        self.cov_inv = jc.sparse.to_dense(self.cov_inv)

        # JIT and vmap versions of mean and jacobian
        self._mean_function_single = jax.jit(self.mean_function)
        self._mean_function_batch = jax.jit(jax.vmap(self.mean_function))
        self._jacobian_function_batch = jax.jit(jax.vmap(jax.jacfwd(self.mean_function)))

        # Precompile with dummy input to avoid compile time later
        dummy_theta = jnp.array([[0.25, 0.8]])
        dummy_sample  = self._mean_function_batch(dummy_theta)
        _ = self._jacobian_function_batch(dummy_theta)
        _ = self.fisher_score(dummy_sample, dummy_theta)

    def mean_function(self, theta):
        ''''
        simulate the angular power spectrum given cosmological parameters theta.
        theta: (2,) array of cosmological parameters, where theta[0] is Omega_c and theta[1] is sigma8.
        returns: (n_ell,) array of angular power spectrum values.

        '''
        cosmo = jc.Planck15(Omega_c=theta[0], sigma8=theta[1])
        m = jc.angular_cl.angular_cl(cosmo, self.ell, self.probes)
        return m.flatten() * self.scalor

    def sample(self, theta):
        '''
        sample from the model given cosmological parameters theta.
        theta: (batch,2) array of cosmological parameters, where theta[:,0] is Omega_c and theta[:,1] is sigma8.
        returns: (N, n_ell) array of samples from N(theta[0], theta[1]) for each ell.   
        '''
        means = self._mean_function_batch(theta)
        cov_np = np.array(self.cov)
        means_np = np.array(means)
        rng = np.random.default_rng()
        samples = [rng.multivariate_normal(mean, cov_np) for mean in means_np]
        return np.stack(samples)
    
    @partial(jax.jit, static_argnums=0) 
    def fisher_score(self, x_sample, theta):
        '''
        compute the Fisher score for given samples and parameters using automatic differentiation.
        x_sample: (N, n_ell) array of samples from the model.
        theta: (N, 2) array of cosmological parameters, where theta[:,0] is Omega_c and theta[:,1] is sigma8.
        returns: (N, 2) array of Fisher scores.
        '''
        mu_all = self._mean_function_batch(theta)
        J_all = self._jacobian_function_batch(theta)

        def score_fn(delta, J):
            return jnp.dot(delta, jnp.dot(self.cov_inv, J))

        # return np.array(jax.vmap(score_fn)(x_sample - mu_all, J_all))
        return jax.vmap(score_fn)(x_sample - mu_all, J_all)

    def fisher_matrix_batch(self, theta_batch):
        """
        Compute Fisher matrices for multiple parameter vectors.
        theta_batch: shape (N, 2)
        Returns:
            Fisher matrices of shape (N, 2, 2)
        """
        J_batch = self._jacobian_function_batch(theta_batch)  # (N, d, 2)
        
        def compute_fisher(J):
            return J.T @ self.cov_inv @ J

        fisher_fn = jax.jit(jax.vmap(compute_fisher))
        return np.array(fisher_fn(J_batch))

    def sample_and_score(self, theta):
        '''
        Sample from the model and compute the Fisher score.
        theta: (N, 2) array of cosmological parameters, where theta[:,0] is Omega_c and theta[:,1] is sigma8.
        returns: (N, n_ell) array of samples from the model and (N, 2) array of Fisher scores.
        '''
        x_sample = self.sample(theta)
        score = self.fisher_score(x_sample, theta)
        return x_sample, score
