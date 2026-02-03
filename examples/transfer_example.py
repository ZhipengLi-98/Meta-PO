"""Transfer learning example using Branin benchmark."""

import torch
import math
from meta_pbo import SequentialPBO, TransferPBO


def branin(x, shift_x1=0.0, shift_x2=0.0):
    """
    Branin function - standard 2D optimization benchmark.
    
    Domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    Has 3 global minima at f(x*) ≈ 0.397887
    
    Different shifts create different but related tasks.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    # Scale from [0,1] to Branin domain
    x1 = x[:, 0] * 15 - 5 + shift_x1  # [-5, 10] + shift
    x2 = x[:, 1] * 15 + shift_x2       # [0, 15] + shift
    
    # Branin function
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    
    result = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * torch.cos(x1) + s
    
    return -result  # Negate for maximization


def main():
    print("="*60)
    print("Transfer Learning Example")
    print("="*60)
    
    # Train models on previous tasks (different Branin shifts)
    print("\n[1/3] Training on previous tasks...")
    previous_models = []
    
    task_shifts = [
        (-2.0, 1.0),   # Task 1: shift both dimensions
        (-1.0, 0.5),   # Task 2: moderate shift
        (0.5, -0.5),   # Task 3: opposite shifts
    ]
    
    for i, (shift_x1, shift_x2) in enumerate(task_shifts):
        print(f"\n  Task {i+1}: shift = ({shift_x1:.1f}, {shift_x2:.1f})")
        
        optimizer = SequentialPBO(
            dim=2,
            bounds=[(0, 1), (0, 1)],
            theta_iterations=200,
            gp_iterations=200
        )
        
        objective = lambda x, s1=shift_x1, s2=shift_x2: branin(x, shift_x1=s1, shift_x2=s2)
        optimizer.optimize(n_iterations=8, objective_fn=objective)
        
        # Save model
        optimizer.save_model(f'/tmp/task_{i}.pth')
        previous_models.append(optimizer.model)
        
        print(f"  → Model trained and saved")
    
    # Optimize on new task WITHOUT transfer
    print("\n[2/3] Optimizing new task WITHOUT transfer learning...")
    new_task = lambda x: branin(x, shift_x1=1.0, shift_x2=0.0)
    
    optimizer_no_transfer = SequentialPBO(
        dim=2,
        bounds=[(0, 1), (0, 1)],
        theta_iterations=200,
        gp_iterations=200
    )
    _, regrets_no_transfer = optimizer_no_transfer.optimize(
        n_iterations=15,
        objective_fn=new_task
    )
    
    print(f"  Final regret: {regrets_no_transfer[-1]:.6f}")
    
    # Optimize on new task WITH transfer (TAF-M)
    print("\n[3/3] Optimizing new task WITH transfer learning (TAF-M)...")
    
    optimizer_transfer = TransferPBO(
        dim=2,
        bounds=[(0, 1), (0, 1)],
        previous_models=previous_models,
        transfer_method='taf_m',
        decay_start=5,
        decay_rate=0.1,
        theta_iterations=200,
        gp_iterations=200
    )
    _, regrets_transfer = optimizer_transfer.optimize(
        n_iterations=15,
        objective_fn=new_task
    )
    
    print(f"  Final regret: {regrets_transfer[-1]:.6f}")
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Without transfer: {regrets_no_transfer[-1]:.6f}")
    print(f"With transfer:    {regrets_transfer[-1]:.6f}")
    improvement = (regrets_no_transfer[-1] - regrets_transfer[-1]) / regrets_no_transfer[-1] * 100
    print(f"Improvement:      {improvement:.1f}%")
    
    print("\nRegret curves:")
    print("Iter  No Transfer  With Transfer  Speedup")
    print("-" * 50)
    for i in range(len(regrets_transfer)):
        speedup = regrets_no_transfer[i] / regrets_transfer[i]
        print(f"{i+1:3d}   {regrets_no_transfer[i]:.6f}    {regrets_transfer[i]:.6f}     {speedup:.2f}x")


if __name__ == "__main__":
    main()
