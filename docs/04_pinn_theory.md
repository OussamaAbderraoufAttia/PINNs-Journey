# Module 4: PINN Theory and First Implementation

## The PINN Framework

### Core Idea
Replace the unknown solution `u(x)` with a neural network `u_θ(x)` and enforce the PDE through the loss function.

### Mathematical Formulation

Given PDE: `ℒ[u](x) = f(x)` for `x ∈ Ω`
With BCs: `ℬ[u](x) = g(x)` for `x ∈ ∂Ω`
And ICs: `u(x,0) = u₀(x)` for `x ∈ Ω`

**Neural network approximation**: `u(x) ≈ u_θ(x)`

**Residual**: `R_θ(x) = ℒ[u_θ](x) - f(x)`

**Loss function**:
```
L(θ) = λ_p L_physics + λ_b L_boundary + λ_i L_initial
```

Where:
- `L_physics = (1/N_p) ∑ R_θ(x_p)²`
- `L_boundary = (1/N_b) ∑ (u_θ(x_b) - g(x_b))²`
- `L_initial = (1/N_i) ∑ (u_θ(x_i, 0) - u₀(x_i))²`

---

## 1. Universal Approximation Theorem

**Cybenko (1989)**: A feedforward network with a single hidden layer and sigmoid activation can approximate any continuous function on a compact subset of ℝⁿ.

**Implication**: Neural networks can represent PDE solutions arbitrarily well.

### Practical Considerations
- **Width vs Depth**: Wide networks can approximate, but deep networks are more efficient for compositional functions
- **Activation choice**: Tanh/Sin for smooth functions, ReLU for piecewise linear
- **Initialization**: Critical for deep networks (Xavier, He, SIREN)

---

## 2. Automatic Differentiation vs Numerical Differentiation

| Method | Accuracy | Cost | PINN Use |
|--------|----------|------|----------|
| **Finite Differences** | O(h²) | N evaluations | ❌ |
| **Automatic Diff** | Machine precision | ~3-4x forward | ✅ |
| **Symbolic** | Exact | Expression swell | ❌ |
| **Complex Step** | Machine precision | 1 eval | ⚠️ |

**AD is the enabler** - makes PINNs practical by computing exact PDE residuals.

---

## 3. First PINN: Exponential Decay ODE

### Problem
```
dy/dt = -λy,  y(0) = 1
```
Exact: `y(t) = e^{-λt}`

### Network
```
t ──► [Linear] ──► [Tanh] ──► [Linear] ──► ... ──► [Linear] ──► ŷ(t)
```

### Physics Loss
```
L_phys = (1/N) ∑ (dŷ/dt(t_i) + λŷ(t_i))²
```

### Initial Loss
```
L_init = (ŷ(0) - 1)²
```

### Training
```python
for epoch in range(epochs):
    # Sample collocation points
    t = torch.rand(N, 1) * T
    t.requires_grad_(True)
    
    # Forward
    y = model(t)
    dy_dt = grad(y.sum(), t, create_graph=True)[0]
    
    # Losses
    L_phys = ((dy_dt + lambda*y)**2).mean()
    L_init = (model(torch.zeros(1,1)) - 1)**2
    L = L_phys + L_init
    
    # Backward
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
```

---

## 4. Loss Weighting Strategies

### Fixed Weights
```python
L = 1.0 * L_phys + 1.0 * L_bound + 1.0 * L_init
```

### Adaptive Weights (GradNorm)
Balance gradient magnitudes across loss components.

### NTK-based Weighting
Weight inversely proportional to Neural Tangent Kernel eigenvalues.

### Curriculum Learning
Start with easy losses (IC/BC), gradually increase physics weight.

---

## 5. Common Failure Modes

### 1. Spectral Bias (Frequency Principle)
**NNs learn low frequencies first** (Rahaman et al., 2019)
- High-frequency components (sharp gradients, shocks) learned last
- Solutions: Fourier features, SIREN, progressive training

### 2. Gradient Pathology
- **Vanishing gradients**: Deep networks with tanh/sigmoid
- **Exploding gradients**: High-order derivatives amplify
- Solutions: Residual connections, proper initialization, gradient clipping

### 3. Imbalanced Losses
```
Physics loss: 1e-2
Boundary loss: 1e-6
→ Network ignores boundary!
```
Solutions: Weight tuning, adaptive weighting, loss normalization

### 4. Optimization Difficulty
- Non-convex loss landscape
- Multiple local minima
- Stiff PDEs (multiple time scales)
- Solutions: LBFGS, learning rate schedules, better sampling

---

## 6. Practical Tips for PINN Success

### Architecture
```python
# Good defaults
hidden_layers = [64, 64, 64, 64]  # 4 hidden layers
activation = 'tanh'  # or 'sin' for periodic
init = 'xavier_normal'
```

### Training
```python
# Adam + LBFGS hybrid
optimizer = Adam(lr=1e-3)  # Phase 1: exploration
# ... train ...
optimizer = LBFGS(lr=1.0, max_iter=20)  # Phase 2: convergence
```

### Sampling
```python
# Latin Hypercube or Sobol > Uniform
sampler = qmc.LatinHypercube(d=2)
points = sampler.random(N) * (bounds_max - bounds_min) + bounds_min
```

### Monitoring
```python
# Track ALL losses separately
history = {'physics': [], 'boundary': [], 'initial': [], 'total': []}

# Compute relative L2 error if analytical available
L2_rel = norm(pred - exact) / norm(exact)
```

---

## 7. Theoretical Guarantees

### Convergence (Shin et al., 2020)
Under certain conditions, PINN minimizer converges to PDE solution as:
- Network capacity → ∞
- Training points → ∞
- Optimization finds global minimum

### Error Bounds
```
‖u - u_θ‖ ≤ C₁ ‖R_θ‖ + C₂ (approximation error) + C₃ (optimization error)
```

**Practical implication**: Small residual ≠ small error if network can't represent solution!

---

## 8. Exercises

### Exercise 1: Derive the PINN Loss for Poisson Equation
```
-∇²u = f in Ω
u = g on ∂Ω
```
Write out L_physics, L_boundary.

### Exercise 2: System of ODEs
```
dy/dt = v
dv/dt = -ω²y
```
Network outputs `[y, v]`. Write the physics loss.

### Exercise 3: Inverse Problem
Given observations `{(t_i, y_i)}`, estimate λ in `dy/dt = -λy`.
Add `λ` as learnable parameter. Write total loss.

---

## Summary: PINN Recipe

```python
# 1. Define equation
def residual(model, x):
    u = model(x)
    # compute derivatives via autograd
    return pde_residual(u, x)

# 2. Define losses
L_phys = residual(model, x_colloc).pow(2).mean()
L_bc = (model(x_bc) - g_bc).pow(2).mean()
L_ic = (model(x_ic) - u0).pow(2).mean()

# 3. Train
L = L_phys + L_bc + L_ic
optimizer.zero_grad()
L.backward()
optimizer.step()

# 4. Evaluate
error = relative_L2(model(x_test), u_exact(x_test))
```

---

> **Next**: Module 5 - Heat Equation (First PDE)