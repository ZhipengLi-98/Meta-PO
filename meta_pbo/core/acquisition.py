"""Acquisition functions."""

import torch
from torch.distributions.normal import Normal


def expected_improvement(mu, sigma, f_best, kappa=0.0):
    """
    Expected Improvement acquisition function.
    
    Args:
        mu: Predictive mean (N,)
        sigma: Predictive std (N,)
        f_best: Current best value
        kappa: Exploration parameter
    
    Returns:
        EI values (N,)
    """
    normal = Normal(0, 1)
    z = (mu - f_best + kappa) / sigma
    ei = (mu - f_best + kappa) * normal.cdf(z) + sigma * torch.exp(normal.log_prob(z))
    ei[sigma == 0] = 0
    return ei


def upper_confidence_bound(mu, sigma, beta=4.0):
    """
    Upper Confidence Bound acquisition function.
    
    Args:
        mu: Predictive mean (N,)
        sigma: Predictive std (N,)
        beta: Exploration parameter
    
    Returns:
        UCB values (N,)
    """
    return mu + beta * sigma
