# Module 8: Optimization Challenges and Modern Improvements

---

## 1. Why PINN Training is Hard

### Non-Convex Landscape
- High-dimensional parameter space (10⁴-10⁶ params)
- Multiple loss terms with different scales
- Stiff PDEs create ill-conditioned Hessian

### Spectral Bias (Frequency Principle)
> **Neural networks learn low frequencies first** (Rahaman et al., 2019)

```python
# PINN on high-frequency problem
u(x) = sin(50x)  # Learns slowly!
u(x) = sin(x)    # Learns fast!
```

### Gradient Pathology
```
Physics loss gradient:  ∇_θ ‖R_θ‖²
Boundary loss gradient: ∇_θ ‖u_θ - g‖²

These can point in OPPOSITE directions!
```

---

## 2. Modern Improvements

### 2.1 Activation Functions

| Activation | Formula | PINN Benefit |
|------------|---------|--------------|
| **Adaptive Tanh** | `tanh(n·x)` | Learnable frequency |
| **Sin (SIREN)** | `sin(ω₀·x)` | Exact high-freq derivatives |
| **Swish** | `x·sigmoid(βx)` | Smooth, non-monotonic |
| **GELU** | `x·Φ(x)` | Transformer-style |
| **Periodic** | `Σ a_i sin(ω_i x)` | Explicit frequencies |

### 2.2 Adaptive Activations (Jagtap et al., 2020)
```python
class AdaptiveActivation(nn.Module):
    def __init__(self, base_act=nn.Tanh(), init_n=1.0):
        self.n = nn.Parameter(torch.tensor(init_n))
        self.base = base_act
    
    def forward(self, x):
        return self.base(self.n * x)
```

### 2.3 Fourier Features (Tancik et al., 2020)
```python
# γ(x) = [cos(2πBx), sin(2πBx)], B ~ N(0, σ²I)
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, mapping_size=256, scale=10.0):
        self.B = nn.Parameter(torch.randn(mapping_size, in_dim) * scale, 
                             requires_grad=False)
    
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B.T
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
```

### 2.4 Residual Connections
```python
class ResidualMLP(nn.Module):
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # Skip connection
        return x
```

---

## 3. Loss Balancing Strategies

### 3.1 GradNorm (Chen et al., 2018)
```python
def gradnorm_update(losses, model, alpha=1.5):
    # Compute gradient norms for each loss
    grad_norms = []
    for i, L in enumerate(losses):
        grad = torch.autograd.grad(L, model.shared_params(), 
                                   retain_graph=True)[0]
        grad_norms.append(grad.norm())
    
    # Target: equal gradient norms
    target = torch.stack(grad_norms).mean()
    
    # Update weights: w_i ← w_i * (target / G_i)^α
    for i in range(len(losses)):
        weights[i] *= (target / grad_norms[i]) ** alpha
```

### 3.2 NTK-based Weighting (Wang et al., 2021)
```python
# Weight inversely proportional to NTK eigenvalues
weights = 1 / (ntk_eigenvalues + epsilon)
```

### 3.3 Loss Balancing Heuristic
```python
def balance_losses(loss_dict, initial_losses):
    """Normalize by initial loss values"""
    balanced = {}
    for name, loss in loss_dict.items():
        balanced[name] = loss * (initial_losses['physics'] / initial_losses[name])
    return balanced
```

### 3.4 Curriculum Learning
```python
# Start easy, increase difficulty
for epoch in range(epochs):
    if epoch < 2000:
        weights = {'physics': 0.1, 'bc': 1.0, 'ic': 1.0}
    elif epoch < 5000:
        weights = {'physics': 0.5, 'bc': 1.0, 'ic': 1.0}
    else:
        weights = {'physics': 1.0, 'bc': 1.0, 'ic': 1.0}
```

---

## 4. Optimizers for PINNs

### Adam (Default)
```python
optimizer = Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
# Good for exploration, but slow convergence
```

### LBFGS (For Convergence)
```python
optimizer = LBFGS(
    lr=1.0,
    max_iter=20,
    max_eval=None,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    history_size=100,
    line_search_fn='strong_wolfe'  # Critical!
)
# Use after Adam for final convergence
```

### Hybrid Strategy
```python
# Phase 1: Adam (exploration)
for epoch in range(10000):
    adam_step()

# Phase 2: LBFGS (convergence)
optimizer = LBFGS(lr=1.0)
for epoch in range(2000):
    def closure():
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        return loss
    optimizer.step(closure)
```

### Learning Rate Schedules
```python
# Cosine annealing with warm restarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)

# Reduce on plateau
scheduler = ReduceLROnPlateau(optimizer, patience=500, factor=0.5)

# OneCycle (fast convergence)
scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=epochs)
```

---

## 5. Advanced Sampling

### 5.1 RAR (Residual-based Adaptive Refinement)
```python
def add_rar_points(model, equation, n_add=1000):
    # Generate candidates
    x_cand = sample_domain(n_add * 10)
    
    # Evaluate residuals
    with torch.enable_grad():
        R = equation.residual(model, x_cand)
    
    # Select top-n
    _, indices = torch.topk(R.abs().flatten(), n_add)
    return x_cand[indices]
```

### 5.2 RAR-G (Gaussian Distribution)
```python
def add_rarg_points(model, equation, n_add=1000):
    # Sample from residual distribution
    probs = R.abs().flatten() / R.abs().sum()
    indices = torch.multinomial(probs, n_add, replacement=True)
    return x_cand[indices]
```

### 5.3 Importance Sampling (Nabian et al., 2021)
```python
class ImportanceSampler:
    def __init__(self, model, equation, n_candidates=10000):
        self.model = model
        self.equation = equation
        self.update_distribution(n_candidates)
    
    def update_distribution(self, n_candidates):
        x_cand = sample_domain(n_candidates)
        R = self.equation.residual(self.model, x_cand)
        self.probs = R.abs() / R.abs().sum()
        self.candidates = x_cand
    
    def sample(self, n):
        idx = torch.multinomial(self.probs, n, replacement=True)
        return self.candidates[idx]
```

---

## 6. Architecture Innovations

### 6.1 Multi-Scale PINN (Mscale)
```python
class MultiScalePINN(nn.Module):
    def __init__(self, scales=[1, 10, 100]):
        self.branches = nn.ModuleList([
            MLP(input_dim * len(scales), ...)
            for _ in scales
        ])
    
    def forward(self, x):
        features = torch.cat([torch.sin(s * x) for s in self.scales], dim=-1)
        return self.branches[0](features)  # Or ensemble
```

### 6.2 Multi-Fidelity PINN
```python
class MultiFidelityPINN(nn.Module):
    def __init__(self):
        self.low_fidelity = MLP(...)      # Coarse physics
        self.correction = MLP(...)        # Learn discrepancy
    
    def forward(self, x):
        return self.low_fidelity(x) + self.correction(x)
```

### 6.3 Conservation-Enforcing PINN
```python
def conservation_loss(model, x):
    # Enforce ∫ u dx = constant
    u = model(x)
    integral = u.mean() * domain_volume
    return (integral - target_mass) ** 2
```

---

## 7. Hyperparameter Guidelines

### Network Architecture
| Problem Type | Layers | Neurons | Activation |
|--------------|--------|---------|------------|
| Simple ODE | 3-4 | 32-64 | Tanh |
| Heat/Burgers | 4-6 | 64-128 | Tanh/Sin |
| Wave/Schrödinger | 5-8 | 128-256 | Sin/Swish |
| High-freq/Shocks | 4-6 | 128-256 | Sin/Fourier |
| 3D/High-dim | 6-8 | 256-512 | Fourier/Siren |

### Training
| Parameter | Recommendation |
|-----------|----------------|
| LR (Adam) | 1e-3 → 1e-4 (cosine) |
| LR (LBFGS) | 1.0 |
| Batch size | All points (full batch) |
| Epochs (Adam) | 10k-50k |
| Epochs (LBFGS) | 2k-5k |
| Gradient clip | 1.0 |
| Weight decay | 1e-4 (if overfitting) |

### Loss Weights (Starting Points)
```python
weights = {
    'physics': 1.0,
    'boundary': 10.0,   # Often need 10-100x
    'initial': 10.0,    # Often need 10-100x
    'data': 1.0,        # If available
}
# Then use GradNorm or manual tuning
```

### Sampling
| Strategy | When to Use |
|----------|-------------|
| Uniform | Simple problems, debugging |
| LHS | Most problems (good default) |
| Sobol | Need deterministic, good coverage |
| RAR | Shocks, boundary layers |
| Importance | Complex loss landscapes |

---

## 8. Debugging Checklist

### Loss Not Decreasing
- [ ] Check gradients: `torch.autograd.gradcheck`
- [ ] Verify residual computation with analytical test
- [ ] Reduce LR, add gradient clipping
- [ ] Check loss weights (physics ≫ BC?)

### BCs Not Satisfied
- [ ] Increase BC weight (10x, 100x)
- [ ] Add more boundary points
- [ ] Check BC implementation (signs, locations)
- [ ] Try harder activation (Sin)

### NaN Losses
- [ ] Gradient clipping: `clip_grad_norm_(params, 1.0)`
- [ ] Reduce LR
- [ ] Check for division by zero in residual
- [ ] Use `torch.nan_to_num` as safety

### Different Runs, Different Results
- [ ] Set ALL seeds: `torch.manual_seed`, `np.random.seed`, `random.seed`
- [ ] `torch.use_deterministic_algorithms(True)`
- [ ] `CUBLAS_WORKSPACE_CONFIG=:4096:8`

### Slow Training
- [ ] Use GPU
- [ ] Reduce `retain_graph=True` usage
- [ ] Try `torch.compile(model)` (PyTorch 2.0)
- [ ] Batch points if memory allows

---

## Summary: Modern PINN Pipeline

```python
# 1. Network with modern components
model = MLP(
    hidden_dims=[128, 128, 128, 128],
    activation='sin',          # or 'swish', 'tanh'
    fourier_features=True,     # for high-freq
    scale=10.0,
)

# 2. Adaptive loss weighting
loss_fn = AdaptiveLossWeighting(strategy='gradnorm')

# 3. Hybrid optimizer
optimizer = AdamW(lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(T_0=1000)

# 4. Adaptive sampling
sampler = RARGSampler(model, equation, update_freq=1000)

# 5. Training loop
for epoch in range(epochs):
    # Sample
    x_phys = sampler.sample(n_phys)
    x_bc, x_ic = sample_bc_ic()
    
    # Losses
    losses = compute_losses(model, x_phys, x_bc, x_ic)
    losses = loss_fn.update(losses, model)
    
    # Optimize
    optimizer.zero_grad()
    losses['total'].backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    # Adaptive sampling
    if epoch % 1000 == 0:
        sampler.update_distribution()
    
    # Switch to LBFGS at end
    if epoch == switch_epoch:
        optimizer = LBFGS(lr=1.0, line_search_fn='strong_wolfe')
```

---

> **Key Takeaway**: Modern PINNs combine **architecture improvements** (Sin, Fourier, ResNet) + **training improvements** (LBFGS, adaptive sampling, loss balancing) to solve previously intractable problems.