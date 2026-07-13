# Module 2: Mathematical Foundations

## Prerequisites

This module assumes familiarity with:
- Calculus (derivatives, integrals, chain rule)
- Linear algebra (vectors, matrices, eigenvalues)
- Basic differential equations

---

## 1. Ordinary Differential Equations (ODEs)

### Definition
An ODE relates a function `y(t)` and its derivatives:
```
F(t, y, y', y'', ..., y⁽ⁿ⁾) = 0
```

### Order
The **order** is the highest derivative: `n` for `y⁽ⁿ⁾`

### Linear vs Nonlinear
- **Linear**: `y' + p(t)y = q(t)`
- **Nonlinear**: `y' = y²`, `y'' + sin(y) = 0`

### Initial Value Problem (IVP)
```
y' = f(t, y),  y(t₀) = y₀
```
Existence and uniqueness guaranteed if `f` is Lipschitz continuous (Picard-Lindelöf).

### Example: Exponential Decay
```
dy/dt = -λy,  y(0) = y₀
```
Solution: `y(t) = y₀ e^{-λt}`

---

## 2. Partial Differential Equations (PDEs)

### Definition
A PDE involves partial derivatives of a function `u(x₁, x₂, ..., x_d)`:
```
F(x, u, ∂u/∂x₁, ..., ∂²u/∂xᵢ∂xⱼ, ...) = 0
```

### Classification (Second-Order Linear)

For `A u_xx + 2B u_xy + C u_yy + ... = 0`:

| Type | Condition | Example | Physics |
|------|-----------|---------|---------|
| **Elliptic** | B² - AC < 0 | ∇²u = f | Steady-state, Poisson |
| **Parabolic** | B² - AC = 0 | u_t = α∇²u | Diffusion, Heat |
| **Hyperbolic** | B² - AC > 0 | u_tt = c²∇²u | Waves, Transport |

### Common PDEs in This Course

| Equation | Type | Order | Key Feature |
|----------|------|-------|-------------|
| Heat: `u_t = α u_xx` | Parabolic | 2nd | Smoothing, max principle |
| Wave: `u_tt = c² u_xx` | Hyperbolic | 2nd | Finite speed, energy conservation |
| Poisson: `∇²u = f` | Elliptic | 2nd | Boundary value problem |
| Burgers: `u_t + u u_x = ν u_xx` | Parabolic | 2nd | Nonlinear, shocks |
| Schrödinger: `iψ_t = -½ψ_xx + Vψ` | Schrödinger | 2nd | Complex, unitary |

---

## 3. Boundary and Initial Conditions

### Boundary Conditions (BCs)

For a domain `Ω` with boundary `∂Ω`:

| Type | Mathematical Form | Physical Meaning |
|------|-------------------|------------------|
| **Dirichlet** | `u = g` on `∂Ω` | Fixed value (temperature, displacement) |
| **Neumann** | `∂u/∂n = g` on `∂Ω` | Fixed flux (heat flux, stress) |
| **Robin** | `αu + β∂u/∂n = g` | Convection/radiation |
| **Periodic** | `u(x) = u(x+L)` | Repeating domain |

### Initial Conditions (ICs)

For time-dependent problems:
```
u(x, 0) = u₀(x)  (initial displacement)
u_t(x, 0) = v₀(x)  (initial velocity - wave eq)
```

### Well-Posedness (Hadamard)
A problem is well-posed if:
1. **Existence**: A solution exists
2. **Uniqueness**: The solution is unique
3. **Stability**: Solution depends continuously on data

---

## 4. Strong vs Weak Formulation

### Strong Form
The PDE holds **pointwise**:
```
-∇·(k∇u) = f  in Ω
u = g_D on Γ_D
k∇u·n = g_N on Γ_N
```

### Weak Form
Multiply by test function `v ∈ V` and integrate:
```
∫_Ω k∇u·∇v dx = ∫_Ω f v dx + ∫_{Γ_N} g_N v ds
```

**Why weak form?**
- Allows less regular solutions (Sobolev spaces)
- Natural for finite elements
- Boundary conditions split: Essential (Dirichlet) vs Natural (Neumann)

### PINNs Use Strong Form
PINNs enforce the PDE **pointwise at collocation points**.
This is the "strong form" approach - simpler but requires smooth solutions.

---

## 5. The Residual

For a PDE `ℒ[u] = f`, the **residual** is:
```
R[u] = ℒ[u] - f
```

- **Exact solution**: `R[u_exact] = 0` everywhere
- **Approximate solution**: `R[u_approx] ≠ 0`

### PINN Loss = Residual Minimization
```
L(θ) = ‖R[u_θ]‖² + ‖BCs‖² + ‖ICs‖²
```

### Types of Residuals

| PDE | Residual |
|-----|----------|
| ODE: `y' + λy = 0` | `R = dy/dt + λy` |
| Heat: `u_t - α u_xx = 0` | `R = u_t - α u_xx` |
| Wave: `u_tt - c² u_xx = 0` | `R = u_tt - c² u_xx` |
| Poisson: `-∇²u - f = 0` | `R = -∇²u - f` |
| Burgers: `u_t + u u_x - ν u_xx = 0` | `R = u_t + u u_x - ν u_xx` |

---

## 6. Loss Function Derivation

### From Variational Principle

Many PDEs arise from minimizing an energy functional:
```
E[u] = ∫_Ω (½|∇u|² - fu) dx
```
Euler-Lagrange equation: `δE/δu = 0` → `-∇²u = f`

### PINN as Energy Minimization

For linear PDEs, PINN loss equals energy error:
```
L = ‖R‖² = ‖-∇²u - f‖²
```

For nonlinear PDEs, no energy functional exists, but residual minimization still works.

### Weighted Loss

```
L = λ_phys ‖R‖² + λ_BC ‖BC‖² + λ_IC ‖IC‖² + λ_data ‖u - u_data‖²
```

**Weight selection is critical!** Too large/small weights cause imbalance.

---

## 7. Automatic Differentiation (AD)

### Chain Rule
For `y = f(g(x))`:
```
dy/dx = f'(g(x)) · g'(x)
```

### Forward Mode AD
Compute derivative alongside value:
```
x = x
ẋ = 1
y = f(x)
ẏ = f'(x) · ẋ
```

### Reverse Mode AD (Backpropagation)
1. **Forward pass**: Compute values, store graph
2. **Backward pass**: Propagate gradients from output to inputs

**PyTorch uses reverse mode** - efficient for scalar output (loss) with many inputs (parameters).

### Higher-Order Derivatives
AD computes **exact derivatives** (to machine precision) by applying chain rule repeatedly:
```
d²y/dx² = d/dx (dy/dx)
```

**Key insight**: PINNs need `u_t`, `u_xx`, `u_tt`, etc. - all computed via AD!

---

## 8. Gradient Computation in PyTorch

### Basic Gradient
```python
x = torch.tensor(2.0, requires_grad=True)
y = x**3
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]  # 12.0
```

### Higher-Order
```python
d2y_dx2 = torch.autograd.grad(dy_dx, x, create_graph=True)[0]  # 12.0
```

### Vector Functions (Jacobian)
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.stack([x[0]**2, x[1]**3])
J = torch.autograd.grad(y[0], x)[0]  # First row
# Or loop for full Jacobian
```

### PDE Derivatives
```python
# For u(x,t) from neural network
u = model(torch.cat([x, t], dim=1))

u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

residual = u_t - alpha * u_xx
```

### Key Flags
| Flag | Purpose |
|------|---------|
| `create_graph=True` | Keep graph for higher-order derivatives |
| `retain_graph=True` | Keep graph for multiple backward passes |
| `grad_outputs` | For non-scalar outputs (usually `ones_like`) |

---

## 9. Common Failure Modes (Mathematical View)

| Issue | Mathematical Cause | Symptom |
|-------|-------------------|---------|
| **Spectral Bias** | NNs learn low frequencies first | Slow convergence for high-frequency solutions |
| **Gradient Vanishing** | Deep networks, tanh/sigmoid | Training stalls early |
| **Imbalanced Losses** | Physics loss ≫ BC loss or vice versa | BCs not satisfied, or physics ignored |
| **Stiffness** | Multiple time scales (e.g., ν ≪ 1) | Optimization difficult |
| **Non-uniqueness** | Multiple solutions satisfy PDE | Converges to wrong solution |

---

## Summary: Key Equations for PINNs

```
PDE Residual:     R(x) = ℒ[u_θ](x) - f(x)
Physics Loss:     L_phys = ∑ R(x_i)²
Boundary Loss:    L_BC = ∑ (u_θ(x_b) - g(x_b))²
Initial Loss:     L_IC = ∑ (u_θ(x,0) - u₀(x))²
Total Loss:       L = λ_p L_phys + λ_b L_BC + λ_i L_IC
Optimization:     θ* = argmin_θ L(θ)
```

---

> **Next**: Module 3 - PyTorch Autograd Deep Dive (hands-on derivatives)