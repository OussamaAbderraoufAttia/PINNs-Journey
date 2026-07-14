import sys
sys.path.insert(0, 'src')
import torch
import numpy as np
from pinns import MLP, set_seed
from pinns.utils.derivatives import gradient

set_seed(42)
model = MLP(input_dim=2, output_dim=1, hidden_dims=[32,32,32], activation='tanh')
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training wave equation (1000 epochs)...")
for epoch in range(1000):
    opt.zero_grad()
    x = torch.rand(50, 1)
    t = torch.rand(50, 1)
    xt = torch.cat([x, t], dim=1).requires_grad_(True)
    u = model(xt)
    g = gradient(u, xt)
    u_x, u_t = g[:, 0:1], g[:, 1:2]
    u_xx = gradient(u_x, xt)[:, 0:1]
    u_tt = gradient(u_t, xt)[:, 1:2]
    loss = torch.mean((u_tt - u_xx) ** 2)
    loss.backward()
    opt.step()
    if (epoch + 1) % 200 == 0:
        print(f"  Epoch {epoch+1}: {loss.item():.4e}")

print("Wave equation training done!")
