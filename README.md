# Meta-PBO: Transfer Learning for Preference-Based Bayesian Optimization

Efficient Bayesian optimization using **pairwise preferences** with **transfer learning**.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from meta_pbo import SequentialPBO

# Define your objective function
def objective(x):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return -(x[:, 0] - 0.5)**2 - (x[:, 1] - 0.3)**2

# Run optimization
optimizer = SequentialPBO(dim=2, bounds=[(0, 1), (0, 1)])
best_x, regrets = optimizer.optimize(n_iterations=20, objective_fn=objective)

print(f"Best point: {best_x}")
print(f"Final regret: {regrets[-1]:.6f}")
```

## Examples

```bash
python examples/basic_example.py        # Sequential PBO
python examples/transfer_example.py     # Transfer learning
```

## Transfer Learning

```python
from meta_pbo import TransferPBO, load_models

# Load previous task models
previous_models = load_models(['task1.pth', 'task2.pth'])

# Optimize with transfer learning
optimizer = TransferPBO(
    dim=2,
    bounds=[(0, 1), (0, 1)],
    previous_models=previous_models,
    transfer_method='taf_m'  # or 'taf_r'
)

best_x, regrets = optimizer.optimize(n_iterations=20, objective_fn=objective)
```

## Key Features

- **Preference-based**: Learn from pairwise comparisons, not absolute values
- **Transfer learning**: TAF-M (variance-based) and TAF-R (ranking-based)
- **Two-step acquisition**: Generate candidates → sample plane → select

## Citation

```bibtex
@inproceedings{li2025efficient,
  title={Efficient Visual Appearance Optimization by Learning from Prior Preferences},
  author={Li, Zhipeng and Liao, Yi-Chi and Holz, Christian},
  booktitle={Proceedings of the 38th Annual ACM Symposium on User Interface Software and Technology},
  pages={1--23},
  year={2025}
}
```

## License

MIT License
