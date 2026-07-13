# Module 5: Heat Equation - Detailed Theory

## Problem Statement

### 1D Heat Equation
```
∂u/∂t = α ∂²u/∂x²,  x ∈ [0, L], t ∈ [0, T]
```

### Boundary Conditions (Dirichlet)
```
u(0, t) = 0
u(L, t) = 0
```

### Initial Condition
```
u(x, 0) = sin(πx/L)
```

### Analytical Solution (Separation of Variables)
```
u(x, t) = e^{-α(π/L)²t} sin(πx/L)
```

---

## 1. Physics of the Heat Equation

### Properties
- **Parabolic**: Information propagates infinitely fast
- **Smoothing**: Initial discontinuities instantly smoothed
- **Maximum Principle**: Max/min occur at boundaries or initial time
- **Energy Dissipation**: `d/dt ∫ u² dx ≤ 0`

### Characteristic Time Scale
```
τ = L²/α
```
For L=1, α=0.01: τ = 100 (but our T=1, so early time behavior)

---

## 2. PINN Formulation

### Domain
```
(x, t) ∈ [0, 1] × [0, 1]
```

### Network
```
u_θ: R² → R
(x, t) ↦ u(x, t)
```

### Residual
```
R(x, t) = ∂u_θ/∂t - α ∂²u_θ/∂x²
```

### Loss Components

**Physics Loss** (collocation points):
```
L_phys = (1/N_p) ∑_{i=1}^{N_p} [u_t(x_i, t_i) - α u_xx(x_i, t_i)]²
```

**Boundary Loss** (x=0 and x=1):
```
L_bc = (1/N_b) ∑ [u_θ(0, t_j)² + u_θ(1, t_j)²]
```

**Initial Loss** (t=0):
```
L_init = (1/N_i) ∑ [u_θ(x_k, 0) - sin(πx_k)]²
```

### Total Loss
```
L = λ_p L_phys + λ_b L_bc + λ_i L_init
```

---

## 3. Sampling Strategies

### Uniform Random
```python
x = torch.rand(N, 1)
t = torch.rand(N, 1)
```
- Simple but can leave gaps
- **Clustering** near boundaries often helps

### Latin Hypercube Sampling (LHS)
```python
from scipy.stats import qmc
sampler = qmc.LatinHypercube(d=2)
samples = sampler.random(N)
x = samples[:, 0:1]
t = samples[:, 1:2]
```
- Better space-filling
- Reduces variance of Monte Carlo estimates

### Sobol Sequences
```python
sampler = qmc.Sobol(d=2, scramble=True)
samples = sampler.random(N)
```
- Low-discrepancy sequence
- Deterministic, good coverage

### Boundary-Focused Sampling
```python
# More points near boundaries where gradients are large
x = torch.rand(N, 1)**2  # Clusters near 0
# Or use importance sampling
```

---

## 4. Loss Weighting for Heat Equation

### Typical Weights
```python
weights = {
    'physics': 1.0,
    'boundary': 10.0,  # Often need stronger BC enforcement
    'initial': 10.0,   # Initial condition critical
}
```

### Why Stronger BC/IC?
- Physics loss has many points (N_p ~ 10,000)
- BC/IC have fewer points (N_b, N_i ~ 1,000)
- Without weighting, physics dominates

### Adaptive Weighting
```python
# Normalize by initial loss values
weights['boundary'] = L_phys_init / L_bc_init
weights['initial'] = L_phys_init / L_init_init
```

---

## 5. Activation Functions for Heat Equation

| Activation | Performance | Reason |
|------------|-------------|--------|
| **Tanh** | Good | Smooth, bounded, good for diffusion |
| **Sin** | Better for long time | Captures oscillatory decay |
| **Swish** | Good | Self-gated, no saturation |
| **ReLU** | Poor | Non-smooth, bad for second derivatives |

### Recommendation: **Tanh** or **Sin**

---

## 6. Network Architecture

### Depth vs Width
```python
# Shallow and wide
[128, 128, 128]  # 3 layers

# Deep and narrow  
[64, 64, 64, 64, 64]  # 5 layers
```
**Deep often better** for PINNs - more expressive for function composition.

### SIREN for Heat Equation
```python
# SIREN with w0=30
# Good for smooth periodic solutions
```

### Fourier Features
```python
# Random Fourier features for high-frequency content
# γ(x) = [cos(2πBx), sin(2πBx)], B ~ N(0, σ²I)
```

---

## 7. Training Strategies

### Phase 1: Adam (Exploration)
```python
optimizer = Adam(lr=1e-3)
for epoch in range(10000):
    # Standard training
```

### Phase 2: LBFGS (Convergence)
```python
optimizer = LBFGS(lr=1.0, max_iter=20, line_search_fn='strong_wolfe')
for epoch in range(2000):
    def closure():
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        return loss
    optimizer.step(closure)
```

### Learning Rate Scheduling
```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
# Or ReduceLROnPlateau
```

---

## 8. Evaluation Metrics

### Relative L2 Error
```
L2_rel = ‖u_θ - u_exact‖₂ / ‖u_exact‖₂
```

### Maximum Error (L∞)
```
L∞ = max |u_θ - u_exact|
```

### Physics Residual Statistics
```python
residual = u_t - α u_xx
mean_res = residual.mean()
max_res = residual.max()
```

---

## 9. Common Issues and Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| BCs not satisfied | Weak boundary weight | Increase λ_b, add more boundary points |
| Slow convergence | Spectral bias | Use Sin activation, Fourier features |
| Oscillations in solution | Too high LR, no gradient clip | Reduce LR, add gradient clipping |
| NaN losses | Exploding gradients | Gradient clipping, smaller LR |
| Different runs, different results | Non-deterministic | Set all seeds, use deterministic algorithms |

---

## 10. Extensions

### 1. Inverse Problem: Find α
```python
alpha = nn.Parameter(torch.tensor(0.1))  # Learnable!
# Add data loss from temperature measurements
```

### 2. 2D Heat Equation
```python
# u_t = α(u_xx + u_yy)
# Residual needs u_xx + u_yy
```

### 3. Variable Coefficients
```python
# u_t = ∇·(α(x)∇u)
# α is function or neural network
```

### 4. Nonlinear Heat Equation
```python
# u_t = ∇·(u^m ∇u)  (Porous medium equation)
```

---

## Summary: Heat Equation PINN Checklist

- [ ] Define analytical solution for verification
- [ ] Implement physics residual: `u_t - α u_xx`
- [ ] Implement Dirichlet BCs at x=0,1
- [ ] Implement IC at t=0
- [ ] Use LHS/Sobol sampling
- [ ] Weight losses appropriately (BC/IC often need 10x)
- [ ] Use Tanh or Sin activation
- [ ] Try 4-6 hidden layers, 64-128 neurons
- [ ] Train Adam → LBFGS
- [ ] Monitor all loss components separately
- [ ] Compute relative L2 error
- [ ] Visualize space-time solution

---

> **Next**: Module 6 - Burgers Equation (Nonlinear Convection-Diffusion)