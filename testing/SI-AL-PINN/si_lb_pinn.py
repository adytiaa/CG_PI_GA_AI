"""
si_lb_pinn.py
PyTorch implementation of SI-LB-PINN (Spatial-Information + Loss-Balancing PINN)
Based on the paper Research on the thermal-fluid coupling in the growth process of Czochralski silicon single crystals based on an improved physics-informed neural network
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import math
import os

# -------------------------
# Config / Hyperparameters
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 2  # x,y
output_dim = 4  # u, v, T, p
hidden_layers = 5
hidden_units = 128
activation = torch.tanh
lr = 0.003
decay_gamma = 0.9
decay_step = 1000
n_epochs = 5000
print_every = 200

# Physics parameters (nondimensionalized or choose accordingly)
nu = 1e-3  # kinematic viscosity
kappa = 1e-3  # thermal diffusivity
rho = 1.0

# -------------------------
# Utility: MLP with SI injection
# -------------------------
class SILayer(nn.Module):
    """
    One layer that injects spatial info via learned encoders M and N and gating Z.
    H_{l} = (1 - Z) * M(X) + Z * N(X), where Z depends on previous hidden H_{l-1}
    We implement a residual-style combination that also uses the linear transform of H_{l-1}.
    """
    def __init__(self, in_features, out_features, input_dim):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        # Encoders M and N that take raw input X (coordinates)
        self.M = nn.Sequential(nn.Linear(input_dim, out_features), nn.Tanh())
        self.N = nn.Sequential(nn.Linear(input_dim, out_features), nn.Tanh())
        # Gate Z: depends on previous hidden state
        self.Z = nn.Sequential(nn.Linear(in_features, out_features), nn.Sigmoid())
        self.activation = nn.Tanh()

    def forward(self, H_prev, X):
        # H_prev: [N, in_features]
        # X: [N, input_dim]
        lin_part = self.lin(H_prev)
        Mx = self.M(X)
        Nx = self.N(X)
        Zx = self.Z(H_prev)
        H = (1 - Zx) * Mx + Zx * Nx + lin_part  # combine spatial encoders and linear transform
        return self.activation(H)


class SI_LB_PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_units=128, hidden_layers=5, output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers

        # input linear
        self.input_lin = nn.Linear(input_dim, hidden_units)
        # build SILayers
        self.silayers = nn.ModuleList([SILayer(hidden_units, hidden_units, input_dim) for _ in range(hidden_layers)])
        # output layer to map to u,v,T,p
        self.out = nn.Linear(hidden_units, output_dim)

    def forward(self, x):
        # x: [N, 2]
        H = torch.tanh(self.input_lin(x))
        for layer in self.silayers:
            H = layer(H, x)
        out = self.out(H)
        # split outputs
        u = out[:, 0:1]
        v = out[:, 1:2]
        T = out[:, 2:3]
        p = out[:, 3:4]
        return u, v, T, p


# -------------------------
# PDE residuals (steady incompressible NS + heat)
# -------------------------
def gradients(u, x, order=1):
    """Compute gradients of u wrt x using autograd. Returns list of derivatives per dimension."""
    grads = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    if order == 1:
        return grads
    else:
        # compute second derivatives (laplacian)
        second = []
        for i in range(grads.shape[1]):
            g = grads[:, i: i+1]
            g2 = autograd.grad(g, x, grad_outputs=torch.ones_like(g), create_graph=True, retain_graph=True)[0][:, i:i+1]
            second.append(g2)
        return torch.cat(second, dim=1)


def pde_residuals(model, x):
    """
    Given coordinates x [N,2], compute PDE residuals:
    - continuity: u_x + v_y = 0
    - momentum x: u u_x + v u_y + p_x - nu (u_xx + u_yy) = 0
    - momentum y: u u_x + v u_y + p_y - nu (v_xx + v_yy) = 0
    - energy: u T_x + v T_y - kappa (T_xx + T_yy) = 0
    Returns residuals stacked as [r_cont, r_momx, r_momy, r_energy]
    """
    x.requires_grad_(True)
    u, v, T, p = model(x)
    # gradients
    grads_u = gradients(u, x)  # [N,2] -> [u_x, u_y]
    grads_v = gradients(v, x)
    grads_T = gradients(T, x)
    grads_p = gradients(p, x)
    u_x = grads_u[:, 0:1]; u_y = grads_u[:, 1:2]
    v_x = grads_v[:, 0:1]; v_y = grads_v[:, 1:2]
    T_x = grads_T[:, 0:1]; T_y = grads_T[:, 1:2]
    p_x = grads_p[:, 0:1]; p_y = grads_p[:, 1:2]

    # second derivatives (laplacian)
    # compute second derivatives by differentiating the first derivatives
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    u_yy = autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 1:2]
    v_xx = autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    v_yy = autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y), create_graph=True, retain_graph=True)[0][:, 1:2]
    T_xx = autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    T_yy = autograd.grad(T_y, x, grad_outputs=torch.ones_like(T_y), create_graph=True, retain_graph=True)[0][:, 1:2]

    # continuity
    r_cont = u_x + v_y

    # momentum (convective + pressure gradient - viscous)
    conv_u = u * u_x + v * u_y
    conv_v = u * v_x + v * v_y
    visc_u = nu * (u_xx + u_yy)
    visc_v = nu * (v_xx + v_yy)
    r_momx = conv_u + p_x - visc_u
    r_momy = conv_v + p_y - visc_v

    # energy
    conv_T = u * T_x + v * T_y
    diff_T = kappa * (T_xx + T_yy)
    r_energy = conv_T - diff_T

    return r_cont, r_momx, r_momy, r_energy


# -------------------------
# Training utilities
# -------------------------
def sample_interior(n):
    # sample points in unit square (0,1)x(0,1)
    x = torch.rand(n, 1)
    y = torch.rand(n, 1)
    return torch.cat([x, y], dim=1).to(device)


def sample_boundary(n_per_side=250):
    # square boundary: four sides
    t = torch.rand(n_per_side, 1)
    left = torch.cat([torch.zeros_like(t), t], dim=1)
    right = torch.cat([torch.ones_like(t), t], dim=1)
    bottom = torch.cat([t, torch.zeros_like(t)], dim=1)
    top = torch.cat([t, torch.ones_like(t)], dim=1)
    b = torch.cat([left, right, bottom, top], dim=0).to(device)
    return b


def bc_values(x):
    # example Dirichlet BCs for u,v,T. (p not specified)
    # Left (x=0): u=0, v=0, T=1
    # Right (x=1): u=0, v=0, T=0
    # bottom/top: u=v=0, insulating T gradient=0 (we'll enforce T value for simplicity)
    vals = {}
    vals['u'] = torch.zeros(x.shape[0], 1).to(device)
    vals['v'] = torch.zeros(x.shape[0], 1).to(device)
    # piecewise T
    T = torch.where(x[:, 0:1] < 0.5, torch.ones_like(x[:, 0:1]), torch.zeros_like(x[:, 0:1]))
    vals['T'] = T.to(device)
    return vals


# -------------------------
# Training function
# -------------------------
def train():
    model = SI_LB_PINN(input_dim=input_dim, hidden_units=hidden_units,
                       hidden_layers=hidden_layers, output_dim=output_dim).to(device)

    # adaptive loss weights parametrized as log variance for each loss term
    # order: continuity, momentum_x, momentum_y, energy, bc_u, bc_v, bc_T
    log_vars = nn.Parameter(torch.zeros(7, device=device), requires_grad=True)

    # optimizer: network params + log_vars
    optimizer = torch.optim.Adam(list(model.parameters()) + [log_vars], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)

    n_interior = 6000
    n_bc_side = 250  # 4 sides -> 1000 bc points total

    for epoch in range(1, n_epochs+1):
        model.train()
        optimizer.zero_grad()

        X_int = sample_interior(n_interior)
        r_cont, r_momx, r_momy, r_energy = pde_residuals(model, X_int)

        # PDE losses (MSE)
        L_cont = torch.mean(r_cont**2)
        L_momx = torch.mean(r_momx**2)
        L_momy = torch.mean(r_momy**2)
        L_energy = torch.mean(r_energy**2)

        # boundary
        X_bc = sample_boundary(n_bc_side)
        u_b_pred, v_b_pred, T_b_pred, p_b_pred = model(X_bc)
        bc_vals = bc_values(X_bc)
        L_bcu = torch.mean((u_b_pred - bc_vals['u'])**2)
        L_bcv = torch.mean((v_b_pred - bc_vals['v'])**2)
        L_bct = torch.mean((T_b_pred - bc_vals['T'])**2)

        # adaptive loss balancing using log_vars (negative log-likelihood style)
        loss_terms = torch.stack([L_cont, L_momx, L_momy, L_energy, L_bcu, L_bcv, L_bct])
        precision = torch.exp(-log_vars)  # precision = 1/var
        weighted_losses = precision * loss_terms + log_vars  # up to constant factor (NLL style)
        loss = torch.sum(weighted_losses)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % print_every == 0 or epoch == 1:
            with torch.no_grad():
                print(f"Epoch {epoch}/{n_epochs} | loss {loss.item():.4e} | "
                      f"L_cont {L_cont.item():.4e} L_momx {L_momx.item():.4e} L_momy {L_momy.item():.4e} L_energy {L_energy.item():.4e}")
                print(" log_vars:", (log_vars.detach().cpu().numpy()).round(4))

    # Save model
    os.makedirs("model_out", exist_ok=True)
    torch.save(model.state_dict(), "model_out/si_lb_pinn.pt")
    print("Saved model to model_out/si_lb_pinn.pt")


if __name__ == '__main__':
    train()
