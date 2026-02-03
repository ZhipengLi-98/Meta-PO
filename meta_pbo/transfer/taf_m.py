"""Transfer Acquisition Function - Mean (TAF-M)."""

import torch
from ..core.acquisition import expected_improvement


class TAF_M:
    """
    TAF-M: Weights previous models by prediction variance.
    Lower variance = higher confidence = higher weight.
    
    Args:
        previous_models: List of trained GP models
        decay_start: Iteration to start decay
        decay_rate: Decay rate per iteration
    """
    
    def __init__(self, previous_models, decay_start=5, decay_rate=0.1):
        self.previous_models = previous_models
        self.decay_start = decay_start
        self.decay_rate = decay_rate
        self.iteration = 0
    
    def __call__(self, x, current_model, f_best, train_means=None):
        """
        Compute TAF-M value.
        
        Args:
            x: Input points (N x D)
            current_model: Current GP model
            f_best: Best observed value
            train_means: Previous models' max training means
        
        Returns:
            Acquisition values (N,)
        """
        # Get variances from all models (with gradients for current model)
        variances = []
        for model in self.previous_models:
            with torch.no_grad():
                variances.append(model(x).variance.detach())
        
        # Current model variance (with gradients)
        variances.append(current_model(x).variance)
        
        variances = torch.stack(variances)
        
        # Inverse variance weighting
        weights = 1.0 / (variances + 1e-6)
        weights = weights / weights.sum(dim=0, keepdim=True)
        
        # Apply decay to previous models
        if self.iteration >= self.decay_start:
            decay = max(0, 1 - (self.iteration - self.decay_start) * self.decay_rate)
            weights[:-1] = weights[:-1] * decay
            weights = weights / weights.sum(dim=0, keepdim=True)
        
        # Compute acquisition
        acq = torch.zeros(x.shape[0], device=x.device, requires_grad=True)
        acq_sum = 0
        
        # Previous models contribution (no gradient needed)
        for i, model in enumerate(self.previous_models):
            with torch.no_grad():
                mean = model(x).mean
                if train_means is not None:
                    contrib = torch.clamp(mean - train_means[i], min=0.0)
                else:
                    contrib = torch.clamp(mean, min=0.0)
                acq_sum = acq_sum + weights[i].detach() * contrib
        
        # Current model contribution (with gradients)
        pred = current_model(x)
        ei = expected_improvement(pred.mean, pred.variance.sqrt(), f_best)
        acq_sum = acq_sum + weights[-1] * ei
        
        return acq_sum
    
    def step(self):
        """Increment iteration counter."""
        self.iteration += 1
