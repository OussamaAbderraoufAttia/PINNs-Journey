# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/username/pinns-journey.git
cd pinns-journey

# Create conda environment
conda env create -f environment.yml
conda activate pinns-journey

# Or install with pip
pip install -r requirements.txt
pip install -e .
```

## Running the Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

Open `notebooks/00_quickstart.ipynb` to begin.

## First PINN Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from src.pinns.utils.derivatives import gradient

# 1. Define the ODE: dy/dt = -y, y(0) = 1
# Exact solution: y(t) = exp(-t)

# 2. Neural network
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

# 3. Physics loss
def physics_loss(model, t):
    t.requires_grad_(True)
    y = model(t)
    dy_dt = gradient(y.sum(), t)[0]
    return torch.mean((dy_dt + y)**2)

# 4. Initial condition loss
def ic_loss(model):
    t0 = torch.zeros(1, 1)
    y0 = model(t0)
    return (y0 - 1.0)**2

# 5. Training
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5000):
    # Sample collocation points
    t_colloc = torch.rand(100, 1) * 2  # t in [0, 2]
    
    loss = physics_loss(model, t_colloc) + ic_loss(model)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4e}")

# 6. Evaluate
model.eval()
with torch.no_grad():
    t_test = torch.linspace(0, 2, 100).unsqueeze(1)
    y_pred = model(t_test)
    y_exact = torch.exp(-t_test)
    error = torch.norm(y_pred - y_exact) / torch.norm(y_exact)
    print(f"Relative L2 Error: {error.item():.4e}")
```

## Project Structure

```
pinns-journey/
├── README.md
├── requirements.txt
├── environment.yml
├── setup.py
├── .gitignore
├── LICENSE
├── docs/                    # Documentation
│   ├── 01_sci_ml_overview.md
│   ├── 02_math_foundations.md
│   ├── ...
├── notebooks/               # Interactive tutorials
│   ├── 00_quickstart.ipynb
│   ├── 01_autograd_fundamentals.ipynb
│   ├── 02_first_pinn_ode.ipynb
│   ├── 03_heat_equation.ipynb
│   ├── 04_burgers_equation.ipynb
│   └── ...
├── src/pinns/              # Core library
│   ├── config.py           # Configuration management
│   ├── models/             # Neural network architectures
│   ├── equations/          # PDE/ODE definitions
│   ├── losses/             # Physics-informed losses
│   ├── sampling/           # Collocation point samplers
│   ├── training/           # Training loop, optimizers
│   ├── evaluation/         # Metrics, visualization
│   └── utils/              # Seeding, logging, derivatives
├── experiments/            # Experiment scripts
├── figures/                # Generated plots
├── papers/                 # Key papers
├── references/             # BibTeX references
└── tests/                  # Unit tests
```

## Learning Path

| Module | Notebook | Theory Doc | Description |
|--------|----------|------------|-------------|
| 1 | - | `01_sci_ml_overview.md` | SciML landscape |
| 2 | - | `02_math_foundations.md` | PDE/ODE theory |
| 3 | `01_autograd_fundamentals.ipynb` | `03_autograd_deep_dive.md` | PyTorch autograd |
| 4 | `02_first_pinn_ode.ipynb` | `04_pinn_theory.md` | First PINN (ODE) |
| 5 | `03_heat_equation.ipynb` | `05_heat_equation.md` | Heat equation (PDE) |
| 6 | `04_burgers_equation.ipynb` | `06_burgers_equation.md` | Burgers (nonlinear) |
| 7 | `05_wave_equation.ipynb` | `07_advanced_pdes.md` | Wave, Poisson, etc. |
| 8 | `06_experiments.ipynb` | `08_optimization_challenges.md` | Modern improvements |
| 9 | `07_operator_learning.ipynb` | `09_modern_improvements.md` | Neural operators |
| 10 | - | `10_operator_learning.md` | DeepONet, FNO |

## Configuration

Create YAML configs in `configs/`:

```yaml
# configs/heat_equation.yaml
name: "heat_1d_experiment"
seed: 42
device: "auto"

model:
  hidden_layers: [64, 64, 64, 64]
  activation: "tanh"
  initialization: "xavier_normal"
  input_dim: 2
  output_dim: 1

training:
  epochs: 10000
  learning_rate: 1e-3
  optimizer: "adam"
  scheduler: "cosine"
  gradient_clip: 1.0
  early_stopping_patience: 1000

sampling:
  n_collocation: 10000
  n_boundary: 1000
  n_initial: 1000
  strategy: "lhs"

equation:
  equation_type: "heat_1d"
  parameters:
    alpha: 0.01

loss_weights:
  physics: 1.0
  boundary: 10.0
  initial: 10.0

logging:
  log_dir: "logs"
  log_freq: 100
  use_wandb: false
```

Run with:
```bash
python -m experiments.run_experiments --config configs/heat_equation.yaml
```

## Common Issues

### NaN Loss
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Reduce learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

### BCs Not Satisfied
```python
# Increase boundary weight
weights = {'physics': 1.0, 'boundary': 100.0, 'initial': 10.0}

# Or use hard constraints
def forward(self, x):
    # Enforce u(0)=0, u(1)=0 by construction
    return x * (1-x) * self.network(x)
```

### Slow Convergence
```python
# Try Sin activation
model = MLP(..., activation='sin')

# Or Fourier features
model = FourierFeatureMLP(scale=10.0)

# Use LBFGS after Adam
optimizer = optim.LBFGS(model.parameters(), lr=1.0, line_search_fn='strong_wolfe')
```

### Different Results Each Run
```python
# Set all seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
```

## Next Steps

1. **Run all notebooks** in order
2. **Read theory docs** in `docs/`
3. **Try experiments** in `experiments/`
4. **Modify configs** for new problems
5. **Read papers** in `papers/`
6. **Contribute** improvements!

## Getting Help

- Check `docs/` for detailed explanations
- Look at `tests/` for usage examples
- Search issues on GitHub
- Read the referenced papers