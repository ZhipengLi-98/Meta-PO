"""Core components for preference-based optimization."""

from .gp_model import ExactGPModel
from .preference import optimize_preference_model
from .kernels import ard_rbf_kernel, ard_matern52_kernel
from .acquisition import expected_improvement, upper_confidence_bound
from .utils import normalize, extend_plane, sample_rectangle, check_bounds

__all__ = [
    'ExactGPModel',
    'optimize_preference_model',
    'ard_rbf_kernel',
    'ard_matern52_kernel',
    'expected_improvement',
    'upper_confidence_bound',
    'normalize',
    'extend_plane',
    'sample_rectangle',
    'check_bounds',
]
