"""Utility functions."""

import torch


def normalize(values, maximum, minimum, scale=1.0):
    """
    Normalize values to [0, scale] range.
    
    Args:
        values: Values to normalize
        maximum: Maximum value
        minimum: Minimum value
        scale: Output scale
    
    Returns:
        Normalized values
    """
    if isinstance(scale, (int, float)):
        scale = torch.tensor(scale, device=values.device)
    return scale * (values - minimum) / (maximum - minimum)


def check_bounds(points, bounds):
    """
    Filter points within bounds.
    
    Args:
        points: List of tensors or tensor (N x D)
        bounds: List of (lower, upper) tuples
    
    Returns:
        Valid points
    """
    if isinstance(points, list):
        points = torch.stack(points)
    
    device = points.device
    lower = torch.tensor([b[0] for b in bounds], device=device)
    upper = torch.tensor([b[1] for b in bounds], device=device)
    
    valid = torch.all((points >= lower) & (points <= upper), dim=-1)
    return points[valid]


def extend_plane(x1, x2, x3, factor=1.25):
    """
    Extend plane defined by 3 points.
    
    Args:
        x1: Center point (D,)
        x2: Second point (D,)
        x3: Third point (D,)
        factor: Extension factor
    
    Returns:
        Four extended corner points
    """
    c = x1
    x2_mirror = 2 * c - x2
    x3_mirror = 2 * c - x3
    
    return (
        c + factor * (x2 - c),
        c + factor * (x2_mirror - c),
        c + factor * (x3 - c),
        c + factor * (x3_mirror - c)
    )


def sample_rectangle(p1, p2, p3, p4, num_x=5, num_y=5):
    """
    Sample points in rectangle.
    
    Args:
        p1, p2, p3, p4: Corner points
        num_x: Samples in x direction
        num_y: Samples in y direction
    
    Returns:
        Sampled points (num_x * num_y, D)
    """
    points = []
    for i in range(num_x):
        for j in range(num_y):
            u = i / max(num_x - 1, 1)
            v = j / max(num_y - 1, 1)
            
            # Bilinear interpolation
            point = (1-u)*(1-v)*p1 + u*(1-v)*p2 + (1-u)*v*p3 + u*v*p4
            points.append(point)
    
    return torch.stack(points)
