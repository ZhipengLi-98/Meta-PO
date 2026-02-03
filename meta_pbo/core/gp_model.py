"""Gaussian Process model for preference learning."""

import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Exact GP with ARD RBF kernel.
    
    Args:
        train_x: Training inputs (N x D)
        train_y: Training outputs (N,)
        likelihood: GPyTorch likelihood
        ard_num_dims: Dimensionality for ARD kernel
    """
    
    def __init__(self, train_x, train_y, likelihood, ard_num_dims=None):
        super().__init__(train_x, train_y, likelihood)
        
        if ard_num_dims is None:
            ard_num_dims = train_x.shape[-1]
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
