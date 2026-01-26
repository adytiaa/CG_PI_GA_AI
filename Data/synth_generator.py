import torch
import numpy as np

def generate_cz_synthetic_data(n_samples=1000):
    """
    Generates synthetic (r, z) coordinates and corresponding (ur, vz, vt, T, p).
    The flow mimics a toroidal vortex driven by rotation and buoyancy.
    """
    # 1. Random sampling in normalized domain [0, 1]
    r = np.random.rand(n_samples, 1)
    z = np.random.rand(n_samples, 1)
    
    # 2. Synthetic Temperature Field (Hot at bottom/sides, Cold at top-center)
    # T = T_melt + deltaT * (r^2 - z) 
    T = 1685.0 + 20.0 * (r**2 + (1-z)) 

    # 3. Synthetic Swirl Velocity (vt)
    # Linearly interpolate between crucible rotation (-0.5) and crystal rotation (1.0)
    omega_crystal = 1.0
    omega_melt = -0.5
    vt = r * (omega_melt * (1 - z) + omega_crystal * z)

    # 4. Poloidal Flow (ur, vz) using a Stream Function to ensure continuity
    # Psi = r^2 * (1-r) * z * (1-z) -> describes a closed loop vortex
    # ur = -(1/r) * dPsi/dz
    # vz =  (1/r) * dPsi/dr
    ur = -r * (1 - r) * (1 - 2 * z)
    vz = (2 * r - 3 * r**2) * z * (1 - z)

    # 5. Synthetic Pressure (Higher at bottom due to gravity/hydrostatics)
    p = 1.0 - 0.1 * z + 0.05 * r**2

    # Convert to Tensors
    X_data = torch.tensor(np.hstack([r, z]), dtype=torch.float32)
    U_data = torch.tensor(np.hstack([ur, vz, vt, T, p]), dtype=torch.float32)
    
    return X_data.to(device), U_data.to(device)

# --- Integration Example ---
# X_syn, U_syn = generate_cz_synthetic_data(2000)
# train(X_syn, U_syn)
