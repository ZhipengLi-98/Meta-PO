"""Transfer Learning Preference-Based Bayesian Optimization."""

import torch
import gpytorch
from typing import List, Tuple, Callable, Optional

from .sequential_pbo import SequentialPBO
from ..transfer import TAF_M, TAF_R


class TransferPBO(SequentialPBO):
    """
    Transfer PBO: Leverages previous task models.
    
    Args:
        dim: Input dimensionality
        bounds: Bounds per dimension
        previous_models: List of trained GP models
        transfer_method: 'taf_m' or 'taf_r'
        decay_start: Start decay at this iteration
        decay_rate: Decay rate per iteration
        **kwargs: Additional SequentialPBO arguments
    """
    
    def __init__(
        self,
        dim: int,
        bounds: List[Tuple[float, float]],
        previous_models: List,
        transfer_method: str = 'taf_m',
        decay_start: int = 5,
        decay_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(dim, bounds, **kwargs)
        
        self.previous_models = previous_models
        self.transfer_method = transfer_method
        
        # Initialize TAF
        if transfer_method == 'taf_m':
            self.taf = TAF_M(previous_models, decay_start, decay_rate)
        elif transfer_method == 'taf_r':
            self.taf = TAF_R(previous_models, decay_start, decay_rate)
        else:
            raise ValueError(f"Unknown method: {transfer_method}")
        
        self.train_means = None
    
    def _update_model(self, x_plus, x_ei, x_ucb, x_chosen):
        """Update model and TAF weights."""
        super()._update_model(x_plus, x_ei, x_ucb, x_chosen)
        
        # Update train means for TAF
        self.train_means = self._compute_train_means()
        
        # Update TAF-R weights if needed
        if self.transfer_method == 'taf_r' and len(self.X) >= 4:
            self.taf.update_weights(torch.stack(self.X), self.model)
    
    def _compute_train_means(self):
        """Compute max training means from previous models."""
        if len(self.X) == 0:
            return None
        
        train_x = torch.stack(self.X)
        means = []
        for model in self.previous_models:
            with torch.no_grad():
                means.append(torch.max(model(train_x).mean))
        
        return torch.stack(means) if means else None
    
    def _optimize_acquisition(self):
        """Optimize TAF to get next candidates."""
        with torch.no_grad():
            f_best = torch.max(self.model(torch.stack(self.X)).mean)
        
        # Optimize TAF
        x_opt = torch.rand((20, self.dim), requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([x_opt], lr=1e-2)
        
        lower = torch.tensor([b[0] for b in self.bounds], device=self.device)
        upper = torch.tensor([b[1] for b in self.bounds], device=self.device)
        
        for _ in range(40):
            optimizer.zero_grad()
            
            # Compute TAF
            taf_values = self.taf(x_opt, self.model, f_best, self.train_means)
            loss = -taf_values.sum()
            
            # Only backward if loss requires grad
            if loss.requires_grad:
                loss.backward()
            
            optimizer.step()
            
            with torch.no_grad():
                x_opt.data = torch.clamp(x_opt.data, lower, upper)
        
        with torch.no_grad():
            taf_values = self.taf(x_opt, self.model, f_best, self.train_means)
            best_idx = taf_values.argmax()
            x_ei = x_opt[best_idx].detach()
            x_ucb = x_opt[best_idx].detach()
        
        self.taf.step()  # Increment iteration
        
        return x_ei, x_ucb


def load_models(model_paths: List[str], device='cpu'):
    """
    Load previous models from checkpoints.
    
    Args:
        model_paths: Paths to .pth files
        device: Device to load on
    
    Returns:
        List of loaded models
    """
    from ..core import ExactGPModel
    
    models = []
    for path in model_paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        train_x = checkpoint['train_x'].to(device)
        train_y = checkpoint['train_y'].to(device)
        dim = train_x.shape[-1]
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPModel(train_x, train_y, likelihood, dim).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        
        if 'length_scale' in checkpoint:
            model.covar_module.base_kernel.lengthscale = checkpoint['length_scale'][:dim]
        
        model.eval()
        likelihood.eval()
        
        models.append(model)
    
    return models
