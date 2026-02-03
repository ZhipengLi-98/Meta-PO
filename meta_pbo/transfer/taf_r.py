"""Transfer Acquisition Function - Ranking (TAF-R)."""

import torch
from ..core.acquisition import expected_improvement


class TAF_R:
    """
    TAF-R: Weights previous models by ranking alignment.
    Models with better preference agreement get higher weight.
    
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
        self.weights = None
    
    def update_weights(self, preference_data, current_model):
        """
        Update weights based on ranking alignment.
        
        Args:
            preference_data: Preference comparisons (N x D)
                            Format: [x1, x2, x3, x_chosen] per group
            current_model: Current GP model
        """
        aligned_counts = []
        
        for prev_model in self.previous_models:
            aligned = 0
            
            # Check alignment for each preference group
            for i in range(0, len(preference_data), 4):
                batch = preference_data[i:i+4]
                
                with torch.no_grad():
                    prev_pred = prev_model(batch).mean
                    curr_pred = current_model(batch).mean
                
                # Count agreements: chosen (idx 3) vs others
                for j in range(3):
                    same_order = (
                        (prev_pred[3] > prev_pred[j] and curr_pred[3] > curr_pred[j]) or
                        (prev_pred[3] < prev_pred[j] and curr_pred[3] < curr_pred[j])
                    )
                    aligned += int(same_order)
            
            aligned_counts.append(aligned)
        
        # Compute weights from alignment
        total = sum(aligned_counts)
        if total > 0:
            self.weights = torch.tensor(aligned_counts, dtype=torch.float32) / total
        else:
            self.weights = torch.ones(len(self.previous_models)) / len(self.previous_models)
        
        # Apply decay
        if self.iteration >= self.decay_start:
            decay = max(0, 1 - (self.iteration - self.decay_start) * self.decay_rate)
            self.weights *= decay
        
        # Add current model weight
        current_weight = torch.tensor([1.0 - self.weights.sum()])
        self.weights = torch.cat([self.weights, current_weight])
        self.weights = self.weights / self.weights.sum()
    
    def __call__(self, x, current_model, f_best, train_means=None):
        """
        Compute TAF-R value.
        
        Args:
            x: Input points (N x D)
            current_model: Current GP model
            f_best: Best observed value
            train_means: Previous models' max training means
        
        Returns:
            Acquisition values (N,)
        """
        if self.weights is None:
            # Initialize uniform weights
            n = len(self.previous_models) + 1
            self.weights = torch.ones(n) / n
        
        acq_sum = 0
        
        # Previous models contribution (no gradient)
        for i, model in enumerate(self.previous_models):
            with torch.no_grad():
                mean = model(x).mean
                if train_means is not None:
                    contrib = torch.clamp(mean - train_means[i], min=0.0)
                else:
                    contrib = torch.clamp(mean, min=0.0)
                acq_sum = acq_sum + self.weights[i].item() * contrib
        
        # Current model contribution (with gradients)
        pred = current_model(x)
        ei = expected_improvement(pred.mean, pred.variance.sqrt(), f_best)
        acq_sum = acq_sum + self.weights[-1].item() * ei
        
        return acq_sum
    
    def step(self):
        """Increment iteration counter."""
        self.iteration += 1
