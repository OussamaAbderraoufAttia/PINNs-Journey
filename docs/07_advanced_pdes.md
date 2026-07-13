# Module 7: Advanced PDEs

---

## 1. Wave Equation

### PDE
```
∂²u/∂t² = c² ∇²u
```

### 1D Wave Equation
```
u_tt = c² u_xx
```

### Conditions
```
u(0, t) = u(1, t) = 0          # Fixed ends
u(x, 0) = sin(πx)              # Initial displacement
u_t(x, 0) = 0                  # Initial velocity
```

### Analytical Solution
```
u(x, t) = sin(πx) cos(cπt)
```

### PINN Challenges
- Second-order time derivative → need `create_graph=True` twice
- Energy conservation: PINNs don't naturally conserve energy
- High-frequency oscillations difficult (spectral bias)

### Residual
```python
u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
residual = u_tt - c**2 * u_xx
```

---

## 2. Poisson Equation (Steady-State)

### PDE
```
-∇²u = f  in Ω
u = g on ∂Ω
```

### 1D Poisson
```
-u_xx = f(x),  u(0)=u(1)=0
```

### Analytical Example
```
f(x) = π² sin(πx)  →  u(x) = sin(πx)
```

### PINN Formulation
- **No time dimension** - simpler!
- Only spatial collocation points
- Boundary loss critical

### Residual
```python
u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
residual = -u_xx - f(x)
```

---

## 3. Reaction-Diffusion Equations

### General Form
```
u_t = D ∇²u + f(u)
```

### Common Reaction Terms

| Name | f(u) | Behavior |
|------|------|----------|
| Fisher-KPP | r u (1-u) | Traveling waves |
| Allen-Cahn | u - u³ | Phase separation |
| Gray-Scott | -u v² + F(1-u) | Patterns |
| Brusselator | A - (B+1)u + u²v | Oscillations |

### PINN Challenges
- Stiffness: fast reaction vs slow diffusion
- Pattern formation: high-frequency modes
- Multiple stable states

---

## 4. Schrödinger Equation (Optional)

### Time-Dependent Schrödinger
```
i ψ_t = -½ ψ_xx + V(x) ψ
```

### Complex-Valued Solution
- Split into real/imaginary: `ψ = u + iv`
- Two coupled real PDEs:
```
u_t = -½ v_xx + V v
v_t =  ½ u_xx - V u
```

### PINN Approach
```python
# Output 2 channels: [ψ_real, ψ_imag]
# Or use complex tensors (PyTorch 1.11+)
```

### Conservation Laws
- Probability: ∫|ψ|² dx = 1 (should be conserved)
- Energy: ⟨H⟩ = constant

---

## 5. Coupled Systems

### Navier-Stokes (Incompressible)
```
u_t + u·∇u = -∇p + ν∇²u
∇·u = 0
```

### PINN Challenges
- Velocity-pressure coupling
- Divergence-free constraint
- High dimensionality

### Approaches
- **Projection methods**: Solve pressure Poisson
- **Penalty methods**: Add ∇·u penalty
- **Stream function**: 2D only

---

## 6. Moving Boundaries / Free Boundaries

### Stefan Problem (Phase Change)
```
u_t = α u_xx  in liquid/solid
u = u_melt  at interface
ρL v_n = k_s ∂u/∂n|_s - k_l ∂u/∂n|_l
```

### PINN Approach
- Level set for interface
- Additional loss for interface conditions
- Adaptive sampling near interface

---

## 7. High-Dimensional PDEs

### Curse of Dimensionality
- Grid methods: O(N^d) points
- PINNs: O(N) points (theoretically)

### Examples
- Black-Scholes (finance): 5-10D
- Hamilton-Jacobi-Bellman: 10-100D
- Fokker-Planck: 2d (state space)

### PINN Advantage
Mesh-free, scales better with dimension

### Challenge
- Optimization harder in high-D
- Need more collocation points
- Spectral bias worse

---

## 8. Parametric PDEs

### Problem
Solve for many parameter values:
```
ℒ(θ)[u] = f,  θ ∈ Θ
```

### Traditional
- Solve separately for each θ
- Or build reduced basis (POD)

### Neural Operators (Better)
- Learn mapping θ ↦ u_θ
- DeepONet, FNO, PINO

---

## 9. Multi-Fidelity PINNs

### Idea
Combine low-fidelity (cheap, inaccurate) and high-fidelity (expensive, accurate) data.

### Loss
```
L = L_HF + λ L_LF + λ_phys L_phys
```

### Applications
- CFD: Coarse mesh + fine mesh
- Multi-scale: Homogenization + DNS

---

## 10. Uncertainty Quantification

### Bayesian PINNs
- Place prior on weights
- Variational inference or MCMC
- Get posterior predictive distribution

### Ensemble PINNs
```python
models = [PINN() for _ in range(10)]
# Train with different seeds/data subsets
# Uncertainty = variance across ensemble
```

### Dropout at Test Time
```python
# Use dropout during inference
# Multiple forward passes → uncertainty
```

---

## Summary: Advanced PDE Capabilities

| PDE | Difficulty | Key PINN Challenge | Best Approach |
|-----|------------|-------------------|---------------|
| Wave | Medium | Energy conservation | Sin activation, IC for u_t |
| Poisson | Easy | BC enforcement | Strong BC weight |
| Reaction-Diffusion | Hard | Stiffness, patterns | Implicit time, adaptive |
| Schrödinger | Hard | Complex, conservation | Split real/imag |
| Navier-Stokes | Very Hard | Coupling, div-free | Penalty/projection |
| High-D | Medium | Optimization | Fourier features |
| Parametric | Hard | Generalization | Neural operators |

---

> **Key Takeaway**: As PDEs get more complex, **vanilla PINNs struggle**. Modern improvements (adaptive activation, Fourier features, neural operators, multi-fidelity) are essential for real applications.

> **Next**: Module 8 - Optimization Challenges & Modern Improvements