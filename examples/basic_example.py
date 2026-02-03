"""Basic example of Sequential PBO."""

import torch
from meta_pbo import SequentialPBO


def simple_2d_function(x):
    """
    Simple 2D function to optimize.
    Global optimum at (0.5, 0.3).
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return -(x[:, 0] - 0.5)**2 - (x[:, 1] - 0.3)**2


def main():
    print("="*60)
    print("Sequential Preference-Based Optimization Example")
    print("="*60)
    
    # Create optimizer
    optimizer = SequentialPBO(
        dim=2,
        bounds=[(0, 1), (0, 1)],
        kernel='rbf',
        btl_s=0.01,
        theta_iterations=200,
        gp_iterations=200
    )
    
    print("\nOptimizing 2D function: -(x-0.5)^2 - (y-0.3)^2")
    print("Target optimum: (0.5, 0.3)")
    print("\nRunning 10 iterations...")
    
    # Run optimization
    best_x, regrets = optimizer.optimize(
        n_iterations=10,
        objective_fn=simple_2d_function
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Best point found: ({best_x[0]:.4f}, {best_x[1]:.4f})")
    print(f"Target optimum:   (0.5000, 0.3000)")
    print(f"\nFinal regret: {regrets[-1]:.6f}")
    print(f"Initial regret: {regrets[0]:.6f}")
    print(f"Improvement: {regrets[0] - regrets[-1]:.6f}")
    
    print("\nRegret history:")
    for i, r in enumerate(regrets):
        print(f"  Iteration {i+1:2d}: {r:.6f}")


if __name__ == "__main__":
    main()
