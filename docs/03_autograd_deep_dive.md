# Module 3: PyTorch Automatic Differentiation Deep Dive

## Why Autograd Matters for PINNs

Every PDE residual requires derivatives:
- `u_t` (time derivative)
- `u_x`, `u_xx` (spatial gradients and Laplacian)
- `u_tt`, `u_xxxx` (higher-order for wave, beam equations)

**PyTorch's `torch.autograd` computes these exactly** using reverse-mode AD.

---

## 1. Computational Graph

```
x (leaf) ──► x² ──► x² + 1 ──► sin(x²+1) ──► y (output)
                │           │
                ▼           ▼
             node       node
```

When you call `y.backward()`, PyTorch:
1. Traverses graph backward
2. Applies chain rule at each node
3. Accumulates gradients in `x.grad`

---

## 2. Key Functions

### `torch.autograd.grad()`
**Most important for PINNs** - computes gradient of any tensor w.r.t. any other.

```python
# Gradient of scalar y w.r.t. tensor x
grad = torch.autograd.grad(
    outputs=y,           # scalar or tensor
    inputs=x,            # tensor with requires_grad=True
    grad_outputs=None,   # for non-scalar outputs
    create_graph=False,  # keep graph for higher-order
    retain_graph=False,  # keep graph after backward
    only_inputs=True     # only compute for inputs
)
# Returns tuple of gradients
```

### `tensor.backward()`
Standard backprop for scalar loss:
```python
loss = (y_pred - y_true).pow(2).mean()
loss.backward()  # Computes dloss/dparams for all params
```

---

## 3. First Derivatives

### Scalar Input, Scalar Output
```python
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1

dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(dy_dx)  # tensor(8.)  = 2*3 + 2
```

### Vector Input, Scalar Output
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x[0]**2 + x[1]**3  # scalar

grad = torch.autograd.grad(y, x, create_graph=True)[0]
print(grad)  # tensor([2., 12.])  = [2*1, 3*2²]
```

### Batch of Inputs
```python
x = torch.randn(10, 2, requires_grad=True)  # 10 points, 2D
y = (x[:, 0]**2 + x[:, 1]**2).sum()  # Sum to scalar

grad = torch.autograd.grad(y, x, create_graph=True)[0]
print(grad.shape)  # torch.Size([10, 2])
```

---

## 4. Second Derivatives (Hessian, Laplacian)

### Second Derivative of Scalar Function
```python
x = torch.tensor(2.0, requires_grad=True)
y = x**3

dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
d2y_dx2 = torch.autograd.grad(dy_dx, x, create_graph=True)[0]
print(d2y_dx2)  # tensor(12.) = 6*2
```

### Laplacian in 2D
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
u = x[0]**2 + x[1]**3

# Gradient
grad_u = torch.autograd.grad(u, x, create_graph=True)[0]

# Laplacian = d²u/dx² + d²u/dy²
laplacian = 0
for i in range(2):
    d2u = torch.autograd.grad(grad_u[i], x, create_graph=True)[0]
    laplacian += d2u[i]

print(laplacian)  # 2 + 6*2 = 14
```

### Utility Function (from our library)
```python
from src.pinns.utils.derivatives import laplacian, gradient, hessian

lap = laplacian(u, x)  # Automatic!
```

---

## 5. Jacobian (Vector-to-Vector)

For `f: Rⁿ → Rᵐ`, Jacobian is `m × n` matrix.

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
f = torch.stack([x[0]**2, x[1]**3, x[0]*x[1]])  # 3 outputs

jacobian = []
for i in range(3):
    grad_i = torch.autograd.grad(f[i], x, create_graph=True, retain_graph=True)[0]
    jacobian.append(grad_i)

J = torch.stack(jacobian)  # Shape: (3, 2)
# [[2*x1, 0], [0, 3*x2²], [x2, x1]] = [[2, 0], [0, 12], [2, 1]]
```

**PyTorch 2.0+**: Use `torch.autograd.functional.jacobian` or `torch.vmap` for efficiency.

---

## 6. PDE Residual Patterns

### Heat Equation: `u_t = α u_xx`
```python
def heat_residual(model, x, t, alpha=0.01):
    xt = torch.cat([x, t], dim=1)
    xt.requires_grad_(True)
    
    u = model(xt)
    
    # u_t
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    
    # u_xx
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    
    return u_t - alpha * u_xx
```

### Wave Equation: `u_tt = c² u_xx`
```python
def wave_residual(model, x, t, c=1.0):
    u = model(torch.cat([x, t], dim=1))
    
    # u_tt
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
    
    # u_xx
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    
    return u_tt - c**2 * u_xx
```

### Burgers Equation: `u_t + u u_x = ν u_xx`
```python
def burgers_residual(model, x, t, nu=0.01):
    u = model(torch.cat([x, t], dim=1))
    
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    
    return u_t + u * u_x - nu * u_xx
```

---

## 7. Common Pitfalls

### Pitfall 1: Forgetting `create_graph=True`
```python
# WRONG - can't compute second derivative
u_x = torch.autograd.grad(u.sum(), x)[0]
u_xx = torch.autograd.grad(u_x.sum(), x)[0]  # ERROR!

# CORRECT
u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
```

### Pitfall 2: Graph Freed After First Backward
```python
# WRONG - graph freed after first grad
g1 = torch.autograd.grad(y1.sum(), x)[0]
g2 = torch.autograd.grad(y2.sum(), x)[0]  # ERROR!

# CORRECT - retain_graph=True
g1 = torch.autograd.grad(y1.sum(), x, retain_graph=True)[0]
g2 = torch.autograd.grad(y2.sum(), x, retain_graph=True)[0]
```

### Pitfall 3: Non-Scalar Output Without `grad_outputs`
```python
# WRONG - y is vector
y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
x = torch.tensor(2.0, requires_grad=True)
z = y * x

grad = torch.autograd.grad(z, x)  # ERROR!

# CORRECT - provide grad_outputs
grad = torch.autograd.grad(z, x, grad_outputs=torch.ones_like(z))
```

### Pitfall 4: In-Place Operations
```python
# WRONG - breaks graph
x = torch.tensor([1.0], requires_grad=True)
x.add_(1)  # In-place!
y = x**2

# CORRECT - out-of-place
x = torch.tensor([1.0], requires_grad=True)
y = (x + 1)**2
```

---

## 8. Memory Management

### Gradient Accumulation
```python
# In training loop, always zero gradients!
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Detaching for Evaluation
```python
# For plotting/evaluation, don't need gradients
with torch.no_grad():
    u_pred = model(x_test)
```

### Retaining Graph Only When Needed
```python
# Only use retain_graph=True when computing multiple gradients
# from the SAME forward pass
u = model(x)
u_x = grad(u.sum(), x, create_graph=True)[0]
u_xx = grad(u_x.sum(), x, create_graph=True)[0]  # No retain needed!

# But if you need u_t AND u_x from same u:
u_t = grad(u.sum(), t, create_graph=True, retain_graph=True)[0]
u_x = grad(u.sum(), x, create_graph=True)[0]  # retain was needed
```

---

## 9. Performance Tips

| Tip | Impact |
|-----|--------|
| Use `sum()` instead of `mean()` for gradients | Avoids division |
| Batch collocation points | GPU utilization |
| Use `torch.vmap` (PyTorch 2.0) for Jacobians | 10-100x faster |
| Minimize `retain_graph=True` | Memory |
| Use `torch.compile()` for model | 20-50% speedup |

### Vectorized Jacobian (PyTorch 2.0)
```python
# Old way: loop
J = []
for i in range(m):
    J.append(grad(y[i], x, create_graph=True)[0])

# New way: vmap
from torch.func import vmap, grad

def f_single(x_i):
    return model(x_i.unsqueeze(0))

J = vmap(grad(f_single))(x_batch)  # Vectorized!
```

---

## 10. Exercises

### Exercise 1: Compute Gradient
```python
# f(x,y) = x²y + y³
# Compute ∇f = [∂f/∂x, ∂f/∂y] at (2, 3)
```

### Exercise 2: Laplacian
```python
# u(x,y) = sin(πx)sin(πy)
# Compute ∇²u = u_xx + u_yy
```

### Exercise 3: Wave Residual
```python
# u(x,t) = sin(πx)cos(πt)
# Compute u_tt - u_xx
```

---

## Summary: Essential Patterns

```python
# 1. First derivative
grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]

# 2. Second derivative
grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
grad2 = torch.autograd.grad(grad.sum(), x, create_graph=True)[0]

# 3. Multiple gradients from same output - use retain_graph
g1 = torch.autograd.grad(y.sum(), x1, create_graph=True, retain_graph=True)[0]
g2 = torch.autograd.grad(y.sum(), x2, create_graph=True)[0]

# 4. PDE residual (heat equation example)
u = model(torch.cat([x, t], dim=1))
u_t = grad(u.sum(), t, create_graph=True)[0]
u_x = grad(u.sum(), x, create_graph=True)[0]
u_xx = grad(u_x.sum(), x, create_graph=True)[0]
residual = u_t - alpha * u_xx
```

---

> **Next**: Module 4 - First PINN (Exponential Decay ODE)