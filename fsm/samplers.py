import numpy as np
from scipy.stats import multivariate_normal

class HamiltonianMonteCarloSampler:
    def __init__(self, score_fn, dim=2, momentum_cov=None, prior=None):
        '''
        a simple implementation of Hamiltonian Monte Carlo sampler.
        score_fn: function that takes a state and returns the score (gradient of log probability)
        dim: dimension of the state space
        momentum_cov: covariance matrix for the momentum distribution, if None, uses identity matrix
        prior: optional, a 2D array with shape (dim, 2) specifying the bounds for each dimension.
               If provided, samples outside these bounds will be rejected.
        '''

        self.score_fn = score_fn
        self.dim = dim 
        self.momentum_cov = momentum_cov if momentum_cov is not None else np.eye(self.dim)
        self.momentum_cov_inv = np.linalg.inv(self.momentum_cov)
        self.momentum_pdf = multivariate_normal(mean=np.zeros(self.dim), cov=self.momentum_cov)
        self.prior = prior

    def sample(self, num_samples, initial_states, leapfrog_steps, leapfrog_step_size, save_trajectory=False):
        """
        Sample from the model using Hamiltonian Monte Carlo.
        initial_states: shape (n_chains, dim)
        num_samples: number of samples to generate
        leapfrog_steps: number of leapfrog steps to take in each iteration
        leapfrog_step_size: step size for the leapfrog integration
        save_trajectory: if True, returns the trajectory of each chain
                         shape (n_chains, num_samples - 1, 2, leapfrog_steps + 1, dim)
        Returns:
            samples: shape (n_chains, num_samples, dim)
            acceptance_rates: shape (n_chains,)
            trajectories: optional, (n_chains, num_samples - 1, 2, leapfrog_steps + 1, dim)
        """
        if initial_states.ndim == 1:
            #add batch dimension if initial_states is a single state
            initial_states = initial_states[np.newaxis, :]

        n_chains, dim = initial_states.shape
        assert dim == self.dim

        samples = np.zeros((n_chains, num_samples, dim))
        samples[:, 0, :] = initial_states
        current_states = initial_states.copy()
        accepted = np.zeros(n_chains)

        if save_trajectory:
            all_traj = []

        for i in range(1, num_samples):
            current_momenta = np.random.multivariate_normal(np.zeros(dim), self.momentum_cov, size=n_chains)
            proposed_states = np.zeros_like(current_states)
            proposed_momenta = np.zeros_like(current_momenta)
            trajs = []

            for c in range(n_chains):
                ps, pm, trac = self.leapfrog(
                    current_states[c].copy(),
                    current_momenta[c].copy(),
                    leapfrog_step_size,
                    leapfrog_steps,
                    save_trajectory
                )
                proposed_states[c] = ps
                proposed_momenta[c] = pm
                if save_trajectory:
                    trajs.append(trac)

            # Apply prior bounds if provided
            if self.prior is not None:
                within_bounds = np.all(proposed_states >= self.prior[:, 0], axis=1) & \
                                np.all(proposed_states <= self.prior[:, 1], axis=1)
            else:
                within_bounds = np.ones(n_chains, dtype=bool)

            # Accept or reject
            accept_mask = within_bounds  # Here we assume acceptance ratio is either 0 or 1
            current_states[accept_mask] = proposed_states[accept_mask]
            accepted += accept_mask.astype(int)

            samples[:, i, :] = current_states

            if save_trajectory:
                all_traj.append(np.stack(trajs))

        if save_trajectory:
            all_traj = np.stack(all_traj)  # shape (num_samples - 1, n_chains, 2, leapfrog_steps + 1, dim)
            all_traj = np.transpose(all_traj, (1, 0, 2, 3, 4))  # (n_chains, num_samples - 1, 2, steps+1, dim)
            return samples, accepted / num_samples, all_traj
        else:
            return samples, accepted / num_samples

    def leapfrog(self, state, momentum, step_size, leapfrog_steps, save_trajectory=False):
        '''
        function to perform leapfrog integration for Hamiltonian Monte Carlo.
        state: current state of the chain, shape (dim,)
        momentum: current momentum of the chain, shape (dim,)
        step_size: step size for the leapfrog integration
        leapfrog_steps: number of leapfrog steps to take
        save_trajectory: if True, returns the trajectory of the chain
        Returns:
            state: new state after leapfrog integration, shape (dim,)
            momentum: new momentum after leapfrog integration, shape (dim,)
            trajectory: optional, trajectory of the chain if save_trajectory is True, shape (2, leapfrog_steps + 1, dim)
        
        '''
        if save_trajectory:
            trajectory = np.zeros((2, leapfrog_steps + 1, len(state)))
            trajectory[0, 0] = state
            trajectory[1, 0] = momentum

        for i in range(leapfrog_steps):
            momentum -= -0.5 * step_size * self.score_fn(state)
            state += step_size * (self.momentum_cov_inv @ momentum)
            momentum -= -0.5 * step_size * self.score_fn(state)

            if save_trajectory:
                trajectory[0, i + 1] = state
                trajectory[1, i + 1] = momentum

        if save_trajectory:
            return state, momentum, trajectory
        else:
            return state, momentum, None
