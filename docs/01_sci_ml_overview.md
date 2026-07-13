# Module 1: Scientific Machine Learning Overview

## The Landscape of Computational Science

Traditionally, scientific computing has been dominated by three distinct fields:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCIENTIFIC COMPUTING                         │
├──────────────────┬──────────────────────┬───────────────────────┤
│                  │                      │                       │
│  NUMERICAL       │   MACHINE LEARNING   │   DATA SCIENCE        │
│  METHODS         │   (TRADITIONAL)      │   / STATISTICS        │
│                  │                      │                       │
│  • FEM/FDM/FVM   │  • Regression        │  • Inference          │
│  • Mesh-based    │  • Classification    │  • Uncertainty        │
│  • Deterministic │  • Pattern recog.    │  • Visualization      │
│                  │                      │                       │
└──────────────────┴──────────────────────┴───────────────────────┘
```

### Scientific Machine Learning (SciML) - The Convergence

**SciML** unifies these fields by using **machine learning tools** (automatic differentiation, optimization, GPUs) to solve **scientific problems** (differential equations, inverse problems, uncertainty quantification).

```
┌─────────────────────────────────────────────────────────────────┐
│                        SCIENTIFIC ML                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  Physics    │    │    Data     │    │   Learning  │        │
│   │  Models     │ +  │  (Sparse/   │ +  │  (Neural    │        │
│   │  (PDEs)     │    │   Noisy)    │    │   Networks) │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                │                 │                    │
│         └────────────────┼─────────────────┘                    │
│                          ▼                                      │
│              ┌─────────────────────────┐                        │
│              │   Physics-Informed      │                        │
│              │   Neural Networks       │                        │
│              └─────────────────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Approaches in SciML

### 1. Physics-Informed Neural Networks (PINNs)
- **Core idea**: Embed PDE residual in loss function
- **Strengths**: Mesh-free, handles inverse problems naturally, differentiable
- **Weaknesses**: Training difficulty, spectral bias, high-dimensional scaling

### 2. Neural Operators (DeepONet, FNO)
- **Core idea**: Learn the *operator* mapping between function spaces
- **Strengths**: Fast inference, generalizes across initial conditions
- **Weaknesses**: Requires lots of training data, limited extrapolation

### 3. Differentiable Simulation
- **Core idea**: Make traditional solvers differentiable
- **Strengths**: High accuracy, physics guarantees
- **Weaknesses**: Mesh-dependent, memory intensive

### 4. Hybrid Methods
- **Examples**: PINN + FEM, PINN + Neural Operators
- **Goal**: Combine strengths of each approach

## PINNs vs Neural Operators vs Traditional Methods

| Aspect | FEM/FDM | PINNs | Neural Operators |
|--------|---------|-------|------------------|
| **Mesh** | Required | No | No |
| **Dimensions** | 1D-3D practical | Any (theoretically) | Any |
| **Inverse Problems** | Difficult | Natural | Possible |
| **Training Data** | None | None (physics) | Required |
| **Inference Speed** | Fast (direct solve) | Slow (eval NN) | Very Fast |
| **Generalization** | Per problem | Per problem | Across ICs/BCs |
| **Accuracy** | High (convergent) | Moderate | Data-dependent |

## When to Use What?

```
                    ┌──────────────────────────────────────┐
                    │         PROBLEM TYPE                 │
                    └──────────────────┬───────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
       ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
       │  FORWARD    │          │  INVERSE    │          │  PARAMETRIC │
       │  (Known IC, │          │  (Sparse    │          │  (Many ICs/ │
       │   BC, want  │          │   data,     │          │   BCs, want │
       │   solution) │          │   find PDE) │          │   fast eval)│
       └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
              │                        │                        │
              ▼                        ▼                        ▼
       ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
       │   FEM/FDM   │          │    PINNs    │          │ Neural Ops  │
       │  (Best)     │          │  (Best)     │          │ (Best)      │
       │  PINNs OK   │          │  Neural Ops │          │  PINNs OK   │
       └─────────────┘          └─────────────┘          └─────────────┘
```

## The PINN Philosophy

> **"Don't just learn the solution. Learn the physics."**

Traditional ML learns a mapping from data. PINNs learn a function that *satisfies physical laws*. This means:
- **Extrapolation** is physically meaningful
- **Conservation laws** are built-in
- **Sparse data** is sufficient
- **Uncertainty** can be quantified through the physics residual

## Roadmap Through This Course

1. **Module 2**: Mathematical Foundations (PDEs, BCs, Weak/Strong forms)
2. **Module 3**: PyTorch Autograd (Derivatives, Jacobians, Hessians)
3. **Module 4**: First PINN (ODE: Exponential Decay)
4. **Module 5**: Heat Equation (PDE: Diffusion)
5. **Module 6**: Burgers Equation (Nonlinear Convection-Diffusion)
6. **Module 7+**: Wave, Poisson, Reaction-Diffusion, Schrödinger
7. **Advanced**: Adaptive sampling, Multi-fidelity, Neural Operators

---

> **Key Takeaway**: PINNs are not a replacement for FEM/FDM. They are a *new tool* for problems where traditional methods struggle: high dimensions, inverse problems, complex geometries, and data assimilation.