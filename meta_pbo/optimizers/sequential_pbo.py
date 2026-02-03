"""Sequential Preference-Based Bayesian Optimization."""

import torch
import gpytorch
from typing import List, Tuple, Callable, Optional

from ..core import (
    ExactGPModel, optimize_preference_model,
    ard_rbf_kernel, expected_improvement,
    normalize, extend_plane, sample_rectangle, check_bounds
)


class SequentialPBO:
    """
    Sequential Preference-Based Bayesian Optimization.
    
    Uses pairwise preferences to optimize without explicit function values.
    Implements two-step acquisition: generate candidates → sample plane → select.
    
    Args:
        dim: Input dimensionality
        bounds: List of (lower, upper) bounds per dimension
        kernel: 'rbf' or 'matern52'
        btl_s: BTL scale parameter (smaller = sharper preferences)
        theta_lr: Learning rate for hyperparameter optimization
        theta_iterations: Iterations for theta optimization
        gp_lr: Learning rate for GP fitting
        gp_iterations: Iterations for GP fitting
        device: 'cpu' or 'cuda'
    """
    
    def __init__(
        self,
        dim: int,
        bounds: List[Tuple[float, float]],
        kernel: str = 'rbf',
        btl_s: float = 0.01,
        theta_lr: float = 1e-3,
        theta_iterations: int = 400,
        gp_lr: float = 1e-3,
        gp_iterations: int = 400,
        device: str = 'cpu'
    ):
        self.dim = dim
        self.bounds = bounds
        self.kernel = kernel
        self.btl_s = btl_s
        self.theta_lr = theta_lr
        self.theta_iterations = theta_iterations
        self.gp_lr = gp_lr
        self.gp_iterations = gp_iterations
        self.device = torch.device(device)
        
        # Optimization state
        self.X = []  # All observed points
        self.goodness = []  # Goodness values for preferences
        self.model = None
        self.likelihood = None
        
        # Kernel function
        from ..core.kernels import ard_rbf_kernel, ard_matern52_kernel
        self.kernel_fn = ard_rbf_kernel if kernel == 'rbf' else ard_matern52_kernel
    
    def optimize(
        self,
        n_iterations: int,
        objective_fn: Callable,
        scale_params: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Run optimization.
        
        Args:
            n_iterations: Number of iterations
            objective_fn: Function to optimize (for simulated preferences)
            scale_params: (maximum, minimum, scale) for normalization
        
        Returns:
            best_x: Best point found
            regrets: Regret history
        """
        # Auto-compute scale if not provided
        if scale_params is None:
            scale_params = self._find_scale_params(objective_fn)
        
        maximum, minimum, scale = scale_params
        regrets = []
        
        # Initialize random points
        x_plus = torch.rand(self.dim, device=self.device)
        x_ei = torch.rand(self.dim, device=self.device)
        x_ucb = torch.rand(self.dim, device=self.device)
        
        for it in range(n_iterations):
            # Generate candidate plane
            p1, p2, p3, p4 = extend_plane(x_plus, x_ei, x_ucb, factor=1.25)
            
            # Sample points on plane
            candidates = [p1, p2, p3, p4, x_plus]
            candidates.extend(sample_rectangle(p1, p2, p3, p4, num_x=5, num_y=5))
            
            # Filter by bounds
            candidates = check_bounds(torch.stack(candidates), self.bounds)
            
            # Simulate preference selection
            values = normalize(objective_fn(candidates), maximum, minimum, scale)
            x_chosen = candidates[torch.argmax(values)]
            
            # Update model
            self._update_model(x_plus, x_ei, x_ucb, x_chosen)
            
            # Compute regret
            regret = (scale - values.max()).item()
            regrets.append(regret)
            
            # Get next candidates
            if it < n_iterations - 1:
                x_ei, x_ucb = self._optimize_acquisition()
                x_plus = x_chosen
        
        # Return best point (minimum regret)
        best_idx = torch.argmin(torch.tensor(regrets))
        best_x = self.X[best_idx * 4 + 3]  # Chosen point from best iteration
        
        return best_x, regrets
    
    def _update_model(self, x_plus, x_ei, x_ucb, x_chosen):
        """Update GP model with new preference."""
        # Add observations
        self.X.extend([x_plus, x_ei, x_ucb, x_chosen])
        self.goodness.extend([1e-6, 1e-6, 1e-6, 1.0])  # x_chosen preferred
        
        X_tensor = torch.stack(self.X).to(self.device)
        
        # Optimize preference model
        theta, g = optimize_preference_model(
            X_tensor, self.goodness, self.dim, self.kernel_fn,
            lr=self.theta_lr, iterations=self.theta_iterations,
            btl_s=self.btl_s
        )
        
        # Fit GP
        train_y = g.to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.likelihood.noise = 1e-4
        self.model = ExactGPModel(X_tensor, train_y, self.likelihood, self.dim).to(self.device)
        
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.gp_lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for _ in range(self.gp_iterations):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        
        # Set lengthscales
        self.model.covar_module.base_kernel.lengthscale = theta[:self.dim]
        self.model.eval()
        self.likelihood.eval()
    
    def _optimize_acquisition(self):
        """Optimize acquisition function to get next candidates."""
        with torch.no_grad():
            f_best = torch.max(self.model(torch.stack(self.X)).mean)
        
        # Optimize EI
        x_opt = torch.rand((20, self.dim), requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([x_opt], lr=1e-2)
        
        lower = torch.tensor([b[0] for b in self.bounds], device=self.device)
        upper = torch.tensor([b[1] for b in self.bounds], device=self.device)
        
        for _ in range(40):
            optimizer.zero_grad()
            pred = self.model(x_opt)
            ei = expected_improvement(pred.mean, pred.variance.sqrt(), f_best)
            loss = -ei.sum()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            with torch.no_grad():
                x_opt.data = torch.clamp(x_opt.data, lower, upper)
        
        with torch.no_grad():
            pred = self.model(x_opt)
            ei = expected_improvement(pred.mean, pred.variance.sqrt(), f_best)
            x_ei = x_opt[ei.argmax()].detach()
            x_ucb = x_opt[ei.argmax()].detach()  # Use same for simplicity
        
        return x_ei, x_ucb
    
    def _find_scale_params(self, objective_fn):
        """Find global min/max for normalization."""
        x_opt = torch.rand((50, self.dim), requires_grad=True, device=self.device)
        
        # Find maximum
        optimizer = torch.optim.Adam([x_opt], lr=1e-2)
        for _ in range(100):
            optimizer.zero_grad()
            values = objective_fn(x_opt)
            loss = -values.sum()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        maximum = objective_fn(x_opt).max()
        
        # Find minimum
        x_opt = torch.rand((50, self.dim), requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([x_opt], lr=1e-2)
        for _ in range(100):
            optimizer.zero_grad()
            values = objective_fn(x_opt)
            loss = values.sum()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        minimum = objective_fn(x_opt).min()
        
        return maximum, minimum, torch.tensor(1.0)
    
    def save_model(self, path: str):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict(),
            'train_x': torch.stack(self.X),
            'train_y': torch.tensor(self.goodness),
            'length_scale': self.model.covar_module.base_kernel.lengthscale,
            'dim': self.dim,
        }, path)
