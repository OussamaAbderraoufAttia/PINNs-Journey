# Module 9: Modern Research - Neural Operators & Beyond

---

## From PINNs to Neural Operators

### PINN Limitation
```
One training = One solution
u_θ solves ℒ[u] = f for ONE (f, BC, IC)
```

### Neural Operator Goal
```
Learn the SOLUTION OPERATOR:  G: (f, BC, IC) ↦ u
One training = Infinite solutions
```

---

## 1. DeepONet (Lu et al., 2021)

### Architecture
```
Input function f  ──► Branch Net  ──► b(f) ∈ R^p
Input coordinates x ──► Trunk Net ──► t(x) ∈ R^p

Output:  u(x) = ⟨b(f), t(x)⟩ = Σ b_i(f) t_i(x)
```

### Universal Approximation Theorem for Operators
> DeepONet can approximate any continuous nonlinear operator.

### Variants
- **Stacked DeepONet**: Multiple trunk/branch pairs
- **PINO**: Physics-informed DeepONet (add PDE residual)

---

## 2. Fourier Neural Operator (FNO) (Li et al., 2021)

### Key Idea
Learn in Fourier space:
```
v_{t+1}(x) = σ(W v_t(x) + (F⁻¹ R F) v_t(x) + b)
```
- `F`: FFT, `F⁻¹`: iFFT
- `R`: Learnable spectral filter (diagonal in Fourier space)
- **Global convolution** via FFT

### Advantages
- **Resolution invariant**: Train on 64×64, evaluate on 256×256
- **Global receptive field**: Each layer sees whole domain
- **Fast**: O(N log N) via FFT vs O(N²) for attention

### Architecture
```
Input u_0(x) ──► Lift to v_0(x) = P u_0(x)
                │
                ▼ (Fourier layers)
            v_T(x)
                │
                ▼
            Project to u(x) = Q v_T(x)
```

---

## 3. Physics-Informed Neural Operators (PINO)

### Loss Function
```
L = L_data + λ_phys L_phys + λ_ic L_ic + λ_bc L_bc
```
- `L_data`: Supervised on solution pairs
- `L_phys`: PDE residual (like PINN but batched)
- Can train on **limited data + physics**

### Applications
- Turbulence modeling
- Weather forecasting
- Inverse problems with few observations

---

## 4. Other Neural Operators

| Method | Key Idea | Strength |
|--------|----------|----------|
| **DeepONet** | Branch/Trunk decomposition | General operators |
| **FNO** | Fourier layers | Periodic, regular grids |
| **PINO** | Physics-informed FNO | Data-scarce, physics |
| **GNO** | Graph neural operators | Irregular domains |
| **MP-PDE** | Message passing | Mesh-based |
| **Neural Green's** | Learn Green's function | Linear PDEs |

---

## 5. Operator Learning vs PINNs

| Aspect | PINNs | Neural Operators |
|--------|-------|------------------|
| **Training** | Per problem | Once for family |
| **Inference** | Slow (forward pass) | Fast (one pass) |
| **Data needed** | None (physics only) | Lots of solution pairs |
| **Generalization** | Poor (new IC/BC) | Good (learned operator) |
| **Resolution** | Fixed | FNO: resolution-invariant |
| **Implementation** | Simple | Complex |

---

## 6. When to Use What?

```
PROBLEM TYPE                    RECOMMENDED APPROACH
────────────────────────────────────────────────────────────
Single forward problem,        PINN (or FEM/FDM)
no data, need interpretability

Many forward solves,           Neural Operator (FNO/DeepONet)
same PDE, varying IC/BC

Sparse data + physics,         PINO or PINN + Data
few observations

Real-time inference,           Neural Operator
parametric studies

Complex geometry,              GNO / Mesh-based
unstructured grids

Inverse problem,               PINN (or PINN + Bayesian)
parameter estimation
```

---

## 7. Hybrid Approaches (Best of Both Worlds)

### 1. PINN Pre-training + Operator Fine-tuning
```python
# 1. Train PINN for specific problem
pinn = train_pinn()

# 2. Generate dataset from PINN
dataset = [(params, pinn_solution) for params in param_samples]

# 3. Train operator on PINN-generated data
operator = train_operator(dataset)
```

### 2. Multi-Fidelity Neural Operators
```python
# Low-fidelity: Coarse FEM
# High-fidelity: DNS / Experiment
# Operator learns correction: u_HF = u_LF + NN(u_LF, params)
```

### 3. Physics-Informed Loss for Operators
```python
def operator_loss(model, batch):
    params, coords, u_true = batch
    u_pred = model(params, coords)
    
    # Data loss
    L_data = MSE(u_pred, u_true)
    
    # Physics loss (differentiate through operator!)
    L_phys = pde_residual(u_pred, coords, params)
    
    return L_data + λ * L_phys
```

---

## 8. Transformer-Based Approaches

### Physics-Informed Transformers
- Attention for long-range interactions
- Positional encoding for coordinates
- Scales to high dimensions

### Example: Galerkin Transformer (Cao et al.)
```
Attention(Q, K, V) with physics-based positional encoding
```

---

## 9. Uncertainty Quantification

### Bayesian PINNs
```python
# Variational inference
q(w) = N(μ, σ²)
ELBO = E_q[log p(data|w)] - KL(q||p)
```

### Ensemble Methods
```python
ensemble = [train_pinn(seed=i) for i in range(10)]
u_mean = torch.stack([m(x) for m in ensemble]).mean(0)
u_std = torch.stack([m(x) for m in ensemble]).std(0)
```

### Conformal Prediction
```python
# Distribution-free uncertainty
calibration_set = get_calibration_data()
# Compute conformal prediction intervals
```

---

## 10. Software Ecosystem

### Frameworks

| Framework | Language | Strengths |
|-----------|----------|-----------|
| **DeepXDE** | Python | Easy PINNs, many BCs |
| **NVIDIA Modulus** | Python | Industrial scale, multi-physics |
| **NeuralOperator** | Python | FNO, DeepONet implementations |
| **JAX/PyTorch** | Python | Custom research |
| **Julia (NeuralPDE)** | Julia | Symbolic, MTK integration |

### Comparison
```
For Learning:     Raw PyTorch (this course!) ✓
For Research:     JAX (faster autograd, vmap)
For Production:   Modulus / DeepXDE
For Operators:    NeuralOperator library
```

---

## 11. Open Research Problems

1. **High-dimensional operators** (>100D)
2. **Long-time integration** stability
3. **Discontinuous solutions** (shocks, interfaces)
4. **Generalization to unseen geometries**
5. **Conservative operators** (mass/momentum/energy)
6. **Foundation models for PDEs** (pre-train on many PDEs)
7. **Inverse problems with operators**
8. **Real-time control with operators**

---

## 12. Recommended Papers (2023-2024)

### Neural Operators
- "Neural Operators for Solving Parametric PDEs" (Kovachki et al., 2023)
- "Operator Learning with Fourier Neural Operators" (Li et al., 2023)
- "Physics-Informed Neural Operators" (Li et al., 2023)

### Improved PINNs
- "Adaptive Activation Functions Accelerate Convergence" (Jagtap et al., 2020)
- "Fourier Features Let Networks Learn High Frequency Functions" (Tancik et al., 2020)
- "Residual-based Adaptive Refinement" (Wu et al., 2022)

### Theory
- "On the Convergence of PINNs" (Shin et al., 2023)
- "Spectral Bias of PINNs" (Wang et al., 2022)
- "Generalization Bounds for PINNs" (Chen et al., 2023)

### Applications
- "Turbulence Modeling with Neural Operators" (Kochkov et al., 2021)
- "Weather Forecasting with FourCastNet" (Pathak et al., 2022)
- "Digital Twins with PINNs" (Haghighat et al., 2023)

---

## Summary: The Evolution

```
1990s:  Neural networks for ODEs/PDEs (Lagaris et al.)
  │
  ▼
2017:   Physics-Informed Neural Networks (Raissi et al.) ← THIS COURSE
  │
  ▼
2020:  Fourier Features, SIREN, Adaptive Activations
  │
  ▼
2021:  Neural Operators (DeepONet, FNO)
  │
  ▼
2022:  PINO, Multi-fidelity, Transformers
  │
  ▼
2023+:  Foundation Models, Conservative Operators, UQ
  │
  ▼
FUTURE:  Scientific Foundation Models for PDEs
```

---

> **Final Thought**: This course gave you the **foundations** (raw PyTorch PINNs). The **modern tools** (DeepXDE, Modulus, NeuralOperator) build on these same principles. Understanding the fundamentals lets you:
> - Debug when libraries fail
> - Extend methods for new physics
> - Read and implement new papers
> - Know when NOT to use PINNs

> **Keep learning, keep coding, keep publishing!** 🚀