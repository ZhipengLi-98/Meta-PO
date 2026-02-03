"""Preference learning using Bradley-Terry-Luce model."""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def compute_btl_likelihood(g, s=0.01):
    """
    Bradley-Terry-Luce preference likelihood.
    
    Args:
        g: Goodness values (N,) where N % 4 == 0
           Format: [x1, x2, x3, x_chosen] for each comparison
        s: BTL scale parameter
    
    Returns:
        Log-likelihood
    """
    log_prob = torch.tensor(0.0, dtype=g.dtype, device=g.device)
    
    for i in range(0, len(g), 4):
        numerator = torch.exp(g[i + 3] / s)
        denominator = sum(torch.exp(g[i + j] / s) for j in range(4))
        log_prob += torch.log(numerator / denominator)
    
    return log_prob


def compute_gp_prior(g, theta, X, kernel_fn):
    """
    GP prior p(g | θ).
    
    Args:
        g: Goodness values (N,)
        theta: Kernel hyperparameters
        X: Input points (N x D)
        kernel_fn: Kernel function
    
    Returns:
        Log probability
    """
    K = kernel_fn(X, X, theta)
    K += 1e-6 * torch.eye(K.shape[0], device=K.device)
    
    mean = torch.zeros(g.size(0), device=g.device)
    
    try:
        mvn = MultivariateNormal(mean, covariance_matrix=K)
        return mvn.log_prob(g)
    except:
        return torch.tensor(-1e10, device=g.device)


def compute_hyperparameter_prior(theta, dim):
    """
    LogNormal prior on hyperparameters.
    
    Args:
        theta: Hyperparameters (dim + 2,)
        dim: Input dimensionality
    
    Returns:
        Log prior probability
    """
    log_prob = torch.tensor(0.0, device=theta.device)
    
    # Lengthscales: LogNormal(ln(0.5), 0.1)
    mu = torch.log(torch.tensor(0.5, device=theta.device))
    sigma = torch.tensor(0.1, device=theta.device)
    
    for i in range(dim):
        if theta[i] > 0:  # Only compute if valid
            log_prob += torch.distributions.LogNormal(mu, sigma).log_prob(theta[i])
        else:
            log_prob += torch.tensor(-1e10, device=theta.device)
    
    # Output scale
    if theta[-2] > 0:
        log_prob += torch.distributions.LogNormal(mu, sigma).log_prob(theta[-2])
    else:
        log_prob += torch.tensor(-1e10, device=theta.device)
    
    return log_prob


def optimize_preference_model(X, initial_goodness, dim, kernel_fn, 
                              lr=1e-3, iterations=400, btl_s=0.01):
    """
    Optimize preference model (theta and g jointly).
    
    Args:
        X: Input points (N x D)
        initial_goodness: Initial goodness values (N,)
        dim: Input dimensionality
        kernel_fn: Kernel function
        lr: Learning rate
        iterations: Optimization iterations
        btl_s: BTL scale parameter
    
    Returns:
        Optimized (theta, g)
    """
    # Initialize parameters in log space for positivity constraint
    log_theta_init = [0.0] * (dim + 2)  # exp(0) = 1.0
    params = log_theta_init + list(initial_goodness)
    params_param = torch.nn.Parameter(torch.tensor(params, device=X.device))
    
    optimizer = torch.optim.Adam([params_param], lr=lr)
    
    for _ in range(iterations):
        optimizer.zero_grad()
        
        # Transform to ensure theta > 0
        log_theta = params_param[:(dim + 2)]
        theta = torch.exp(log_theta)  # Always positive
        g = params_param[(dim + 2):]
        
        # Negative log posterior
        loss = -(
            compute_btl_likelihood(g, btl_s) +
            compute_gp_prior(g, theta, X, kernel_fn) +
            compute_hyperparameter_prior(theta, dim)
        )
        
        if not torch.isfinite(loss):
            # Skip this iteration if loss is not finite
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_param, max_norm=1.0)
        optimizer.step()
    
    # Final transformation
    log_theta = params_param[:(dim + 2)].detach()
    theta = torch.exp(log_theta)
    g = params_param[(dim + 2):].detach()
    
    return theta, g
