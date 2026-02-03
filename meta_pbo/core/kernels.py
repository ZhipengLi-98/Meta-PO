"""ARD kernel functions."""

import torch


def ard_rbf_kernel(X1, X2, theta):
    """
    ARD RBF kernel.
    
    K(x1, x2) = θ_scale * exp(-0.5 * Σ((x1_i - x2_i)^2 / θ_i^2))
    
    Args:
        X1: Points (N x D)
        X2: Points (M x D)
        theta: [lengthscales (D,), output_scale, noise]
    
    Returns:
        Kernel matrix (N x M)
    """
    device = theta.device
    X1, X2 = X1.to(device), X2.to(device)
    
    dim = X1.shape[-1]
    lengthscales = theta[:dim]
    output_scale = theta[-2]
    
    # Scaled squared distances
    dist = torch.sum(
        ((X1.unsqueeze(1) - X2.unsqueeze(0)) ** 2) / (lengthscales ** 2),
        dim=-1
    )
    
    return output_scale * torch.exp(-0.5 * dist)


def ard_matern52_kernel(X1, X2, theta):
    """
    ARD Matérn 5/2 kernel.
    
    K(x1, x2) = (1 + √5*r + 5*r^2/3) * exp(-√5*r)
    
    Args:
        X1: Points (N x D)
        X2: Points (M x D)
        theta: [lengthscales (D,), output_scale, noise]
    
    Returns:
        Kernel matrix (N x M)
    """
    import math
    
    device = theta.device
    X1, X2 = X1.to(device), X2.to(device)
    
    dim = X1.shape[-1]
    lengthscales = theta[:dim]
    
    # Scale and compute distance
    X1_scaled = X1 / lengthscales
    X2_scaled = X2 / lengthscales
    dist = torch.cdist(X1_scaled, X2_scaled, p=2)
    
    # Matérn 5/2 formula
    sqrt5 = math.sqrt(5)
    return (1 + sqrt5 * dist + 5 * dist**2 / 3) * torch.exp(-sqrt5 * dist)
