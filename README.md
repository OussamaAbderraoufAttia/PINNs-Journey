# PINNs Journey: From First Principles to Research-Level Implementation

> A comprehensive educational repository for mastering Physics-Informed Neural Networks (PINNs) through progressive, hands-on learning.

---

## 🎯 Project Overview

This repository is designed as a **graduate-level learning path** for Physics-Informed Neural Networks. Unlike tutorials that simply show you how to use a library, this repository builds everything from **first principles using raw PyTorch**, ensuring you understand the mathematics, the implementation details, and the practical nuances that make PINNs work—or fail.

**Target Audience**: Graduate students, research interns, and ML engineers who want to deeply understand Scientific Machine Learning (SciML).

---

## 🤔 Motivation

Traditional numerical methods (FEM, FDM, FVM) have powered scientific computing for decades. However, they face challenges with:
- High-dimensional problems (curse of dimensionality)
- Complex geometries
- Inverse problems and data assimilation
- Real-time inference requirements

PINNs offer a **mesh-free, differentiable programming approach** that naturally handles:
- Forward and inverse problems in a unified framework
- Incorporation of sparse/noisy data
- Uncertainty quantification
- High-dimensional PDEs

But PINNs are **not magic**—they have failure modes, optimization difficulties, and theoretical limitations. This repository teaches you **when they work, why they fail, and how to fix them**.

---

## 📐 What Are PINNs?

**Physics-Informed Neural Networks** are neural networks trained to satisfy:
1. **Governing equations** (PDEs/ODEs) via automatic differentiation
2. **Boundary conditions** (Dirichlet, Neumann, Robin, periodic)
3. **Initial conditions** (for time-dependent problems)
4. **Observational data** (when available)

The key insight: **Automatic differentiation allows us to compute PDE residuals exactly** (up to machine precision) and use them as a physics-based loss function.

```
Total Loss = λ_physics * L_physics + λ_boundary * L_boundary + λ_initial * L_initial + λ_data * L_data
```

---

## 🧮 Mathematical Intuition

### The Core Idea

Given a PDE: `ℒ[u](x) = f(x)` for `x ∈ Ω`
With BC: `ℬ[u](x) = g(x)` for `x ∈ ∂Ω`
And IC: `u(x,0) = u₀(x)` for `x ∈ Ω`

We approximate `u(x) ≈ u_θ(x)` where `u_θ` is a neural network with parameters `θ`.

The **physics residual** is: `r_θ(x) = ℒ[u_θ](x) - f(x)`

We minimize: `L(θ) = ‖r_θ(x)‖² + ‖ℬ[u_θ](x) - g(x)‖² + ‖u_θ(x,0) - u₀(x)‖²`

**Why this works**: Neural networks are universal function approximators. By enforcing the PDE at collocation points, we constrain the function space to physically valid solutions.

---

## 📁 Repository Structure

```
PINNs-Journey/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment
├── LICENSE                   # MIT License
├── .gitignore                # Git ignore rules
│
├── docs/                     # Theory documentation (markdown tutorials)
│   ├── 01_sci_ml_overview.md
│   ├── 02_math_foundations.md
│   ├── 03_autograd_deep_dive.md
│   ├── 04_pinn_theory.md
│   ├── 05_heat_equation.md
│   ├── 06_burgers_equation.md
│   ├── 07_advanced_pdes.md
│   ├── 08_optimization_challenges.md
│   ├── 09_modern_improvements.md
│   ├── 10_operator_learning.md
│   └── references.md
│
├── notebooks/                # Interactive learning notebooks
│   ├── 01_autograd_fundamentals.ipynb
│   ├── 02_first_pinn_ode.ipynb
│   ├── 03_heat_equation.ipynb
│   ├── 04_burgers_equation.ipynb
│   ├── 05_wave_equation.ipynb
│   ├── 06_poisson_equation.ipynb
│   ├── 07_reaction_diffusion.ipynb
│   ├── 08_experiments_analysis.ipynb
│   └── 09_literature_review.ipynb
│
├── src/pinns/                # Core library (modular, typed, documented)
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py            # Multi-layer perceptron
│   │   ├── activations.py    # Activation functions
│   │   └── initializers.py   # Weight initialization
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── physics.py        # PDE residual losses
│   │   ├── boundary.py       # Boundary condition losses
│   │   ├── initial.py        # Initial condition losses
│   │   └── composite.py      # Combined loss functions
│   ├── sampling/
│   │   ├── __init__.py
│   │   ├── collocation.py    # Collocation point strategies
│   │   ├── boundary.py       # Boundary point sampling
│   │   └── adaptive.py       # Adaptive sampling (RAR, etc.)
│   ├── equations/
│   │   ├── __init__.py
│   │   ├── base.py           # Base PDE class
│   │   ├── ode.py            # ODE examples
│   │   ├── heat.py           # Heat equation
│   │   ├── burgers.py        # Burgers equation
│   │   ├── wave.py           # Wave equation
│   │   ├── poisson.py        # Poisson equation
│   │   └── reaction_diffusion.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training loop
│   │   ├── optimizers.py     # Custom optimizers/schedulers
│   │   └── callbacks.py      # Logging, checkpointing, early stopping
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py        # L2 error, relative error, etc.
│   │   └── visualization.py  # Plotting utilities
│   └── utils/
│       ├── __init__.py
│       ├── seeding.py        # Reproducibility
│       ├── logging.py        # Structured logging
│       └── derivatives.py    # Autograd utilities
│
├── experiments/              # Systematic experiment scripts
│   ├── __init__.py
│   ├── configs/              # Experiment configurations
│   ├── run_experiments.py    # Main experiment runner
│   ├── ablation_studies/     # Ablation study scripts
│   └── hyperparameter_search/
│
├── figures/                  # Generated publication-quality figures
│   ├── loss_curves/
│   ├── solution_surfaces/
│   ├── residual_heatmaps/
│   ├── error_maps/
│   └── comparison_plots/
│
├── papers/                   # Key papers (PDFs or links)
│   ├── pinn_original.pdf
│   ├── adaptive_pinns.pdf
│   ├── xpinns.pdf
│   ├── deeponet.pdf
│   └── fno.pdf
│
├── references/               # BibTeX references and summaries
│   ├── references.bib
│   └── paper_summaries.md
│
└── tests/                    # Unit tests
    ├── __init__.py
    ├── test_derivatives.py
    ├── test_losses.py
    ├── test_boundary_conditions.py
    ├── test_models.py
    └── test_equations.py
```

---

## 🚀 Installation

### Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate pinns-journey
```

### Using Pip
```bash
pip install -r requirements.txt
```

### Development Install
```bash
pip install -e .
```

---

## 📚 Learning Path

### Module 1: Scientific Machine Learning Overview (`docs/01_sci_ml_overview.md`, `notebooks/01_autograd_fundamentals.ipynb`)
- ML vs Numerical Methods vs Scientific Computing vs PINNs vs Neural Operators
- When to use each approach

### Module 2: Mathematical Foundations (`docs/02_math_foundations.md`)
- ODEs/PDEs, Boundary/Initial Conditions
- Weak vs Strong Formulations
- Residual-based Loss Derivation
- Automatic Differentiation Theory

### Module 3: PyTorch Autograd Deep Dive (`docs/03_autograd_deep_dive.md`, `notebooks/01_autograd_fundamentals.ipynb`)
- `dy/dx`, second derivatives, Laplacian, Jacobian, Hessian
- Computational graph mechanics
- Gradient flow through derivatives

### Module 4: First PINN - ODE (`docs/04_pinn_theory.md`, `notebooks/02_first_pinn_ode.ipynb`)
- `dy/dx = -y` with `y(0) = 1`
- Network architecture, loss, training, visualization

### Module 5: Heat Equation (`docs/05_heat_equation.md`, `notebooks/03_heat_equation.ipynb`)
- `u_t = α u_xx`
- Sampling strategies, physics/boundary/initial losses

### Module 6: Burgers Equation (`docs/06_burgers_equation.md`, `notebooks/04_burgers_equation.ipynb`)
- Shock formation, optimization challenges
- Activation function choices

### Advanced Modules (7-10)
- Wave Equation, Poisson Equation, Reaction-Diffusion, Schrödinger
- Modern improvements: Adaptive PINNs, XPINNs, VPINNs
- Operator Learning: DeepONets, FNOs, Neural Operators

---

## 🧪 Experiments

Run systematic experiments:
```bash
# Run all experiments
python experiments/run_experiments.py --all

# Specific ablation study
python experiments/run_experiments.py --study activation_functions

# Hyperparameter search
python experiments/hyperparameter_search/search.py --config experiments/configs/hp_search.yaml
```

Experiments cover:
- Network depth/width
- Activation functions (Tanh, Sin, Swish, ReLU)
- Learning rates and optimizers (Adam, LBFGS)
- Collocation point counts and sampling strategies
- Boundary condition weighting
- Spectral bias analysis

---

## 📊 Visualizations

Publication-quality figures generated automatically:
- Loss curves (total, physics, boundary, initial)
- Solution surfaces (2D/3D)
- Residual heatmaps
- Prediction vs analytical solution
- Error maps (absolute, relative)
- Gradient evolution during training
- Convergence comparisons

---

## 🔬 Future Work

- [ ] **Uncertainty Quantification**: Bayesian PINNs, Ensemble methods
- [ ] **Inverse Problems**: Parameter identification from sparse data
- [ ] **Multi-fidelity PINNs**: Combining low/high fidelity data
- [ ] **Conservation Laws**: Strict mass/momentum/energy conservation
- [ ] **GPU-accelerated sampling**: Adaptive collocation on GPU
- [ ] **Distributed Training**: Multi-GPU PINN training
- [ ] **Real-world Applications**: Fluid dynamics, solid mechanics, biomedical

---

## 📖 References

### Foundational Papers
1. **Raissi, Perdikaris, Karniadakis (2019)** - "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs" *JCP*
2. **Lagaris, Likas, Fotiadis (1998)** - "Artificial neural networks for solving ordinary and partial differential equations" *IEEE Trans. Neural Networks*

### Modern Improvements
3. **Jagtap, Kawaguchi, Karniadakis (2020)** - "Adaptive activation functions accelerate convergence in deep and physics-informed neural networks" *JCP*
4. **Nabian, Gladstone, Meidani (2021)** - "Efficient training of physics-informed neural networks via importance sampling" *NeurIPS*
5. **Cai, Mao, Wang, et al. (2022)** - "Physics-informed neural networks (PINNs) for fluid mechanics: A review" *Acta Mechanica Sinica*

### Operator Learning
6. **Lu, Jin, Pang, et al. (2021)** - "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators" *Nature Machine Intelligence*
7. **Li, Kovachki, Azizzadenesheli, et al. (2021)** - "Fourier neural operator for parametric partial differential equations" *ICLR*

### Comprehensive Reviews
8. **Karniadakis, Kevrekidis, Lu, et al. (2021)** - "Physics-informed machine learning" *Nature Reviews Physics*
9. **Cuomo, Di Cola, Giampaolo, et al. (2022)** - "Scientific machine learning through physics-informed neural networks: Where we are and what's next" *J. Sci. Comput.*

See `references/references.bib` for complete BibTeX entries and `references/paper_summaries.md` for detailed summaries.

---

## 📄 License

MIT License - see `LICENSE` file for details.

---

## 🙏 Acknowledgments

This repository was built as a learning journey. Special thanks to the SciML community for open research and educational resources.

---

**Happy Learning!** 🎓

*Start with `notebooks/01_autograd_fundamentals.ipynb` and `docs/01_sci_ml_overview.md`*