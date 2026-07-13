# Module 6: Burgers Equation - Nonlinear Convection-Diffusion

## Why Burgers Equation?

Burgers equation is the **simplest nonlinear PDE** combining:
- **Convection**: `u u_x` (nonlinear, wave steepening)
- **Diffusion**: `ν u_xx` (smoothing, shock thickness ∝ √ν)

It's a 1D model for:
- Gas dynamics
- Traffic flow
- Turbulence (energy cascade)
- Shallow water waves

---

## 1. Mathematical Formulation

### Viscous Burgers Equation
```
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
```

### Domain & Conditions
```
x ∈ [-1, 1],  t ∈ [0, 1]
u(x, 0) = -sin(πx)           # Initial condition
u(-1, t) = u(1, t) = 0       # Dirichlet boundary conditions
```

### Parameters
```
ν = 0.01/π  (small viscosity → sharp shock)
```

---

## 2. Analytical Solution (Cole-Hopf Transform)

The Cole-Hopf transformation linearizes Burgers:
```
u = -2ν ∂/∂x log(φ)
```
where `φ` satisfies the heat equation:
```
φ_t = ν φ_xx
```

For `u(x,0) = -sin(πx)`, the exact solution involves modified Bessel functions:
```
u(x,t) = 2νπ [I₁(π) sin(πx) exp(-νπ²t)] / [I₀(π) + I₁(π) cos(πx) exp(-νπ²t)]
```
where `I₀`, `I₁` are modified Bessel functions.

**For PINNs**: We use high-resolution finite difference as reference.

---

## 3. PINN Formulation

### Network
```
u_θ: R² → R
(x, t) ↦ u(x, t)
```

### Residual
```
R(x, t) = u_t + u u_x - ν u_xx
```

### Losses
```
L_phys = ⟨R²⟩_collocation
L_bc = ⟨u(-1,t)² + u(1,t)²⟩_boundary
L_init = ⟨u(x,0) + sin(πx)²⟩_initial
L_total = L_phys + L_bc + L_init
```

---

## 4. Why Burgers is Hard for PINNs

### Shock Formation
- Initial condition: smooth sine wave
- Nonlinearity `u u_x` steepens the wave
- Viscosity `ν` prevents true discontinuity
- **Shock thickness**: δ ≈ ν

### Spectral Bias
- PINNs learn low frequencies first
- Shock = high frequency feature
- **Result**: Slow convergence near shock

### Optimization Landscape
- Non-convex loss with many local minima
- Gradient balancing between convection and diffusion terms
- Different time scales: convection (fast) vs diffusion (slow)

---

## 5. Activation Functions Matter

| Activation | Shock Resolution | Training Stability |
|------------|------------------|-------------------|
| Tanh       | Poor             | Good              |
| ReLU       | Poor             | Poor (2nd deriv = 0) |
| **Sin**    | **Best**         | Moderate          |
| **Swish**  | Good             | Good              |
| **GELU**   | Good             | Good              |

### Why Sin/Swish Work Better?
- **Sin**: Periodic, good for high frequencies (SIREN theory)
- **Swish**: Smooth, non-monotonic, self-gated
- Both avoid saturation issues of Tanh

---

## 6. Adaptive Sampling (RAR-G)

### Residual-based Adaptive Refinement with Gaussian Distribution

```python
def adaptive_sampling(model, equation, n_add=1000):
    # 1. Generate candidate points
    x_cand = uniform(-1, 1, n_candidates)
    t_cand = uniform(0, 1, n_candidates)
    
    # 2. Compute residuals
    with torch.enable_grad():
        R = equation.residual(model, x_cand, t_cand)
    
    # 3. Sample proportionally to |R|
    probs = |R| / sum(|R|)
    indices = torch.multinomial(probs, n_add)
    
    # 4. Add to training set
    x_phys = torch.cat([x_phys, x_cand[indices]])
    t_phys = torch.cat([t_phys, t_cand[indices]])
```

### Effect
- Concentrates points near shock
- Reduces total points needed
- Improves shock resolution

---

## 7. Loss Weighting Strategies

### Fixed Weights (Trial & Error)
```python
weights = {
    'physics': 1.0,
    'boundary': 10.0,   # Strong BCs
    'initial': 10.0,    # Strong IC
}
```

### GradNorm (Chen et al., 2018)
Balance gradient norms:
```python
# Target: equal gradient norms for each loss
target_norm = mean(grad_norms)
λ_i = (grad_norm_i / target_norm)^α
```

### NTK-based Weighting (Wang et al., 2021)
```python
# Weight inversely proportional to initial loss
λ_i = L_i(0) / L_i(current)
```

### Simple Heuristic
```python
# Normalize by number of points
weights = {
    'physics': 1.0,
    'boundary': N_phys / N_bound * 0.1,
    'initial': N_phys / N_init * 0.1,
}
```

---

## 8. Network Architecture Tips

### Fourier Feature Mapping
```python
class FourierFeatureMLP(nn.Module):
    def __init__(self, mapping_size=256, scale=10.0):
        self.B = torch.randn(mapping_size, 2) * scale
        # ...
    
    def forward(self, x):
        x_proj = 2π x @ B.T
        return MLP(torch.cat([cos(x_proj), sin(x_proj)], dim=-1))
```

### SIREN (Sine Activation)
```python
# w0=30 for first layer, w0=1 for others
# Special initialization
```

### Multi-scale Architecture
```python
# Branch for low frequencies (Tanh)
# Branch for high frequencies (Sin)
# Combine with learnable gates
```

---

## 9. Training Protocol for Burgers

### Phase 1: Adam (Epochs 0-10000)
```python
optimizer = Adam(lr=1e-3)
scheduler = CosineAnnealingLR(T_max=10000)
```

### Phase 2: LBFGS (Epochs 10000-15000)
```python
optimizer = LBFGS(lr=1.0, max_iter=20, 
                  line_search_fn='strong_wolfe')
```

### Adaptive Sampling Schedule
```python
adapt_freq = 1000
n_add = 500
for epoch in range(epochs):
    if epoch % adapt_freq == 0:
        add_high_residual_points(n_add)
```

---

## 10. Evaluation Metrics

### Relative L2 Error
```
L2_rel = ‖u_PINN - u_FD‖₂ / ‖u_FD‖₂
```

### Shock Location Error
```python
# Find shock position (max gradient)
shock_pred = x[argmax(|u_x|)]
shock_true = x[argmax(|u_FD_x|)]
shock_error = |shock_pred - shock_true|
```

### Residual Norm
```python
L_residual = ‖R‖₂ / ‖u‖₂
```

---

## 11. Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| Shock not captured | More points, Sin activation, adaptive sampling |
| BCs violated | Increase BC weight, more boundary points |
| Training unstable | Gradient clipping, smaller LR, LBFGS |
| Different runs, different results | Set all seeds, deterministic algorithms |
| Slow convergence | Fourier features, SIREN, multi-scale |

---

## 12. Extensions

### 1. Inviscid Burgers (ν = 0)
- True discontinuity (shock)
- PINNs struggle - need entropy conditions
- Use Lax-Friedrichs flux or TVD regularization

### 2. 2D Burgers
```
u_t + u u_x + v u_y = ν(u_xx + u_yy)
v_t + u v_x + v v_y = ν(v_xx + v_yy)
```

### 3. Stochastic Burgers
- Add noise term
- Requires ensemble training

### 4. Inverse Problem: Find ν
```python
nu = nn.Parameter(torch.tensor(0.1))
# Add data loss from sparse measurements
```

---

## Summary: Burgers PINN Checklist

- [ ] Use Sin or Swish activation (not Tanh)
- [ ] Implement adaptive sampling (RAR-G)
- [ ] Weight BC/IC losses 10-100x physics
- [ ] Use 5-6 layers, 64-128 neurons
- [ ] Train Adam → LBFGS
- [ ] Monitor shock position, not just L2 error
- [ ] Compare with high-res FD reference
- [ ] Try Fourier features if shock still blurry

---

> **Key Insight**: Burgers equation exposes the **spectral bias** of neural networks. Overcoming it requires architectural changes (Sin, Fourier features) AND training strategies (adaptive sampling, loss weighting).

> **Next**: Module 7 - Wave Equation (Second-order in time, energy conservation)