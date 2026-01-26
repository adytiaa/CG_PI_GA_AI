import numpy as np
import pandas as pd
import torch

def generate_and_save_cz_data(filename="cz_synthetic_data.csv", n_samples=1000):
    """
    Generates synthetic CZ data based on analytical approximations 
    and saves it to a local CSV file.
    """
    # 1. Create a grid: more points near the boundaries (crystal/crucible)
    # where the SI-layers need more information.
    r = np.random.uniform(0, 1, (n_samples, 1))
    z = np.random.uniform(0, 1, (n_samples, 1))

    # 2. Physics-based synthetic profiles (Cylindrical Coordinates)
    
    # Temperature: Hotter at crucible (r=1, z=0), Cooler at crystal (r<0.4, z=1)
    # T_normalized = (T - T_melt) / deltaT
    T = 0.5 * (r**2 + (1 - z)) 

    # Swirl Velocity (v_theta): Driven by rotation
    # Crystal rotates at +1.0, Crucible at -0.5
    omega_crystal = 1.0
    omega_crucible = -0.5
    # Mask to define crystal radius (e.g., crystal is 40% of crucible radius)
    v_theta = r * (omega_crucible * (1 - z) + omega_crystal * z)

    # Poloidal Flow (u_r, v_z) using a Stream Function (Psi)
    # This ensures Continuity (div V = 0) is roughly satisfied in the data
    # Psi = r^2 * (1-r) * z * (1-z)
    u_r = -r * (1 - r) * (1 - 2 * z)     # -(1/r) * dPsi/dz
    v_z = (2 * r - 3 * r**2) * z * (1 - z) # (1/r) * dPsi/dr

    # Pressure: Hydrostatic + Dynamic approx
    p = (1 - z) + 0.1 * (u_r**2 + v_z**2)

    # 3. Assemble into a DataFrame
    columns = ['r', 'z', 'u_r', 'v_z', 'v_theta', 'T', 'p']
    data = np.hstack([r, z, u_r, v_z, v_theta, T, p])
    df = pd.DataFrame(data, columns=columns)

    # 4. Save to local file
    df.to_csv(filename, index=False)
    print(f"Successfully saved {n_samples} samples to {filename}")

# Generate the file
generate_and_save_cz_data("cz_synthetic_data.csv", n_samples=1000)
