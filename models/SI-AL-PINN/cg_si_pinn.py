import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "nu": 1e-6,          # Kinematic viscosity (Silicon melt ~10^-7 to 10^-6)
    "kappa": 1e-5,       # Thermal diffusivity
    "beta": 1.4e-4,      # Thermal expansion coefficient
    "g": 9.81,           # Gravity
    "T_ref": 1685.0,     # Melting point of Silicon (K)
    "omega_c": 1.0,      # Crystal rotation speed (rad/s)
    "omega_m": -0.5,     # Crucible (melt) rotation speed (rad/s)
    "layers": 5,
    "hidden": 128,
    "lr": 1e-3,
    "epochs": 10000
}

# --- 2. SI-LB-PINN ARCHITECTURE ---

class SILayer(nn.Module):
    """Spatial Information Layer: Injects r, z coordinates into every layer."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.M = nn.Sequential(nn.Linear(2, out_features), nn.Tanh())
        self.N = nn.Sequential(nn.Linear(2, out_features), nn.Tanh())
        self.Z = nn.Sequential(nn.Linear(in_features, out_features), nn.Sigmoid())

    def forward(self, H_prev, coords):
        # coords are [r, z]
        Mx = self.M(coords)
        Nx = self.N(coords)
        Zx = self.Z(H_prev)
        H = (1 - Zx) * Mx + Zx * Nx + self.lin(H_prev)
        return torch.tanh(H)

class CZ_PINN(nn.Module):
    def __init__(self, hidden=128, layers=5):
        super().__init__()
        # Input: r, z | Output: ur, vz, vtheta, T, p
        self.input_lin = nn.Linear(2, hidden)
        self.silayers = nn.ModuleList([SILayer(hidden, hidden) for _ in range(layers)])
        self.output_lin = nn.Linear(hidden, 5)

    def forward(self, x):
        H = torch.tanh(self.input_lin(x))
        for layer in self.silayers:
            H = layer(H, x)
        return self.output_lin(H)

# --- 3. PHYSICS RESIDUALS (CYLINDRICAL NS + ENERGY) ---

def get_residuals(model, coords):
    coords.requires_grad_(True)
    out = model(coords)
    
    ur = out[:, 0:1]     # Radial velocity
    vz = out[:, 1:2]     # Axial velocity
    vt = out[:, 2:3]     # Swirl velocity (theta)
    T  = out[:, 3:4]     # Temperature
    p  = out[:, 4:5]     # Pressure
    
    r = coords[:, 0:1]
    z = coords[:, 1:2]
    r_s = r + 1e-7 # Prevent division by zero at axis

    def grad(phi, wrt):
        return autograd.grad(phi, wrt, grad_outputs=torch.ones_like(phi), 
                             create_graph=True, retain_graph=True)[0]

    # First derivatives
    d_rz = grad(ur, coords); ur_r, ur_z = d_rz[:, 0:1], d_rz[:, 1:2]
    d_rz = grad(vz, coords); vz_r, vz_z = d_rz[:, 0:1], d_rz[:, 1:2]
    d_rz = grad(vt, coords); vt_r, vt_z = d_rz[:, 0:1], d_rz[:, 1:2]
    d_rz = grad(T,  coords); T_r,  T_z  = d_rz[:, 0:1], d_rz[:, 1:2]
    d_rz = grad(p,  coords); p_r,  p_z  = d_rz[:, 0:1], d_rz[:, 1:2]

    # Second derivatives (Laplacians in Cylindrical)
    ur_rr = grad(ur_r, coords)[:, 0:1]; ur_zz = grad(ur_z, coords)[:, 1:2]
    vz_rr = grad(vz_r, coords)[:, 0:1]; vz_zz = grad(vz_z, coords)[:, 1:2]
    vt_rr = grad(vt_r, coords)[:, 0:1]; vt_zz = grad(vt_z, coords)[:, 1:2]
    T_rr  = grad(T_r,  coords)[:, 0:1]; T_zz  = grad(T_z,  coords)[:, 1:2]

    # Continuity: (1/r)d(r*ur)/dr + dvz/dz = 0
    res_cont = ur/r_s + ur_r + vz_z

    # R-Momentum (with Centrifugal force vt^2/r)
    res_mom_r = (ur*ur_r + vz*ur_z - vt**2/r_s) + p_r - config['nu']*(ur_rr + ur_r/r_s + ur_zz - ur/r_s**2)

    # Theta-Momentum (Swirl)
    res_mom_t = (ur*vt_r + vz*vt_z + ur*vt/r_s) - config['nu']*(vt_rr + vt_r/r_s + vt_zz - vt/r_s**2)

    # Z-Momentum (with Boussinesq Buoyancy)
    buoyancy = config['g'] * config['beta'] * (T - config['T_ref'])
    res_mom_z = (ur*vz_r + vz*vz_z) + p_z - config['nu']*(vz_rr + vz_r/r_s + vz_zz) - buoyancy

    # Energy Equation
    res_energy = (ur*T_r + vz*T_z) - config['kappa']*(T_rr + T_r/r_s + T_zz)

    return res_cont, res_mom_r, res_mom_t, res_mom_z, res_energy

# --- 4. DATA HANDLING & SYMMETRY ---

def get_symmetry_loss(model, n_points=200):
    """Enforces axisymmetry at r=0."""
    z_rand = torch.rand(n_points, 1).to(device)
    r_zero = torch.zeros(n_points, 1).to(device)
    coords_axis = torch.cat([r_zero, z_rand], dim=1).requires_grad_(True)
    
    out = model(coords_axis)
    ur = out[:, 0:1]
    vt = out[:, 2:3]
    T = out[:, 3:4]
    vz = out[:, 1:2]
    
    # Boundary Conditions at Centerline (r=0):
    # ur = 0, vt = 0, dT/dr = 0, dvz/dr = 0
    dT_dr = autograd.grad(T, coords_axis, grad_outputs=torch.ones_like(T), create_graph=True)[0][:, 0:1]
    dvz_dr = autograd.grad(vz, coords_axis, grad_outputs=torch.ones_like(vz), create_graph=True)[0][:, 0:1]
    
    loss_sym = torch.mean(ur**2) + torch.mean(vt**2) + torch.mean(dT_dr**2) + torch.mean(dvz_dr**2)
    return loss_sym

# --- 5. TRAINING LOOP WITH LOSS BALANCING (LB) ---

model = CZ_PINN(config['hidden'], config['layers']).to(device)

# Loss balancing weights (Uncertainty Weighting)
# 5 PDE terms + 1 Symmetry term + 1 Data term = 7 log_vars
log_vars = nn.Parameter(torch.zeros(7, device=device))
optimizer = torch.optim.Adam(list(model.parameters()) + [log_vars], lr=config['lr'])

def train(X_data=None, U_data=None):
    """
    X_data: [N, 2] (r, z) from COMSOL
    U_data: [N, 5] (ur, vz, vt, T, p) from COMSOL
    """
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        # 1. Interior PDE Points
        X_int = torch.rand(2000, 2).to(device) # Normalized Domain [0,1]
        res = get_residuals(model, X_int)
        pde_losses = torch.stack([torch.mean(r**2) for r in res]) # 5 terms
        
        # 2. Symmetry Loss
        loss_sym = get_symmetry_loss(model)
        
        # 3. Data-Informed Loss (if COMSOL data is provided)
        if X_data is not None:
            pred_data = model(X_data)
            loss_data = torch.mean((pred_data - U_data)**2)
        else:
            loss_data = torch.tensor(0.0).to(device)

        # 4. Total Balanced Loss
        # formula: sum( exp(-Li) * loss_i + Li )
        all_losses = torch.cat([pde_losses, loss_sym.unsqueeze(0), loss_data.unsqueeze(0)])
        loss = torch.sum(torch.exp(-log_vars) * all_losses + log_vars)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Total Loss: {loss.item():.4f} | Data Loss: {loss_data.item():.6f}")

# --- 6. EXPLANATION OF SYMMETRY & IMPLEMENTATION ---
"""
How to use this code:

1. THE COORDINATE SYSTEM: 
   Unlike the previous notebook, we use (r, z). By omitting theta, we 
   mathematically assume axisymmetry.

2. SYMMETRY AT THE AXIS (r=0):
   In CZ growth, the physics at the center of the crucible must be smooth. 
   The 'get_symmetry_loss' function enforces that the radial and swirl 
   velocities are zero at the center and that gradients of temperature 
   and axial velocity across the center are zero.

3. ROTATION:
   The swirl velocity 'vt' (v_theta) is included in the Momentum R and 
   Momentum Theta equations. To simulate the crystal rotation, you should 
   add a boundary condition on the top surface where vt = omega_c * r.

4. LOSS BALANCING (LB):
   The 'log_vars' are learned parameters. If the Energy residual is 
   much larger than the Continuity residual, the network will automatically 
   tune the weights so the optimizer focuses on the harder physics terms.
"""

if __name__ == "__main__":
    # Example: Run training without external data first (Pure PINN)
    # To use COMSOL data, load your CSV into tensors and pass to train()
    train()