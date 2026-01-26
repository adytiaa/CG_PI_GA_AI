# SI-LB-PINN — architecture & hyperparameters (based on the paper)

**Network topology**

* Base network: fully-connected MLP (feedforward PINN)
* Inputs: spatial coords (x,y). Outputs: (u, v, T, P) (transverse & longitudinal velocities, temperature, pressure). 

**Hidden layers / neurons**

* **5 hidden layers**, **128 neurons per hidden layer**. 

**Activation & optimizer**

* Activation: **tanh**.
* Optimizer: **Adam**. 

**Learning rate schedule**

* Initial learning rate: **0.003**.
* Decay: multiply by **0.9 every 1000 iterations**. 

**Training data / sampling**

* Boundary points: **1000 random points per boundary** (4 boundaries → 4000 boundary points).
* Interior (configuration) points: **6000 LHS (Latin Hypercube Sampling)**.
* **Total training points: 10,000**. 

**Loss composition & adaptive balancing (the “LB” part)**

* Total loss is a weighted sum of PDE residual loss (L_{PDE}), boundary condition loss (L_{BC}), (and data loss if used). The paper models each loss term’s weight via an uncertainty parameter (\varepsilon). 
* Method: treat each output / loss as Gaussian with variance (\varepsilon^2); the negative log-likelihood yields loss terms of the form (\frac{1}{2\varepsilon^2}L + \log\varepsilon). The variances (\varepsilon) (hence the effective weights (\omega=1/(2\varepsilon^2))) are **learned / updated during training** via maximum likelihood (so the network adapts the per-loss weights automatically). Equations (17)–(21) in the paper show this derivation. 

**Spatial-information injection (the “SI” part)**

* They **reintroduce the raw spatial coordinates into each hidden layer** via two learned nonlinear encoders (M) and (N), and an attention/gating style combination (Z_l). Hidden-layer update (paper notation):
  (H_1=\sigma(W_1 X+b_1)),
  (Z_l=\sigma(W_l H_{l-1}+b_l)),
  (H_l=(1-Z_l)\odot M + Z_l\odot N),
  where (M=\sigma(W_M X + b_M)), (N=\sigma(W_N X + b_N)). This preserves spatial identity and boosts expressivity. 

**Other practical details**

* Automatic differentiation used to compute PDE residuals (standard PINN practice). 
* Hardware used in experiments: Intel i5-12600K CPU, 16 GB RAM, NVIDIA RTX 3080 (8 GB VRAM) — useful to estimate compute. 

---



[1]: https://www.researchgate.net/publication/396134849_Research_on_the_thermal-fluid_coupling_in_the_growth_process_of_Czochralski_silicon_single_crystals_based_on_an_improved_physics-informed_neural_network "(PDF) Research on the thermal-fluid coupling in the growth process of Czochralski silicon single crystals based on an improved physics-informed neural network"



Based on the paper **"Research on the thermal-fluid coupling in the growth process of Czochralski silicon single crystals based on an improved physics-informed neural network"** (DOI: 10.1063/5.0271778), your implementation is a solid foundation, but there are critical adjustments needed to move from a general PINN to the specific **CZ-specific SI-LB-PINN** described in the research.

### 1. Code Review & Improvements

#### A. Coordinate System (Cartesian vs. Cylindrical)
**The Issue:** Your code uses Cartesian coordinates $(x, y)$. The paper focuses on the Czochralski process, which is inherently **axisymmetric**.
**The Fix:** You must redefine the PDE residuals in **Cylindrical Coordinates $(r, z)$**. The Laplacian and divergence operators change, and a "swirl" velocity component ($u_\theta$) is usually required because the crucible and crystal rotate.

#### B. The SI-Layer (Spatial Information)
Your implementation of the `SILayer` is correct according to the Wang/Mao architecture often cited in "Improved PINNs." It correctly uses gating to prevent the "spectral bias" (the tendency of NNs to learn low frequencies only).

#### C. Loss Balancing (LB)
Your use of `log_vars` (Uncertainty Weighting) is consistent with the paper's "LB" component. However, the paper often mentions weighting the **Boundary Conditions (BC)** more heavily in early epochs.

---

### 2. Implementation of Axisymmetry & Physics
To match the paper, your `pde_residuals` function needs to include the **Cylindrical Navier-Stokes** terms.

```python
def pde_residuals_cylindrical(model, coords, nu, kappa, beta, g=9.81):
    # coords: [r, z]
    coords.requires_grad_(True)
    out = model(coords)
    # u: radial velocity (v_r), v: axial velocity (v_z), w: swirl (v_theta), T: temp, p: pressure
    u, v, w, T, p = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4], out[:, 4:5]

    # Gradients
    def grad(phi):
        return autograd.grad(phi, coords, grad_outputs=torch.ones_like(phi), create_graph=True)[0]

    d_r_z = grad(u)
    u_r, u_z = d_r_z[:, 0:1], d_r_z[:, 1:2]
    # ... (similarly for v, w, T, p)

    # 1. Continuity (Cylindrical): (1/r)*d(r*u)/dr + dv/dz = 0
    # Equivalent to: u/r + du/dr + dv/dz = 0
    r = coords[:, 0:1]
    # Add a small epsilon to r to avoid division by zero at the axis
    r_safe = r + 1e-6
    res_cont = u/r_safe + u_r + v_z

    # 2. Momentum Equations (Simplified for brevity)
    # Notice the centrifugal term (w^2 / r) and buoyancy (beta * g * deltaT)
    res_mom_r = (u*u_r + v*u_z - w**2/r_safe) + p_r - nu*(Laplacian(u) - u/r_safe**2)
    
    # 3. Energy Equation with Boussinesq Approximation
    res_energy = (u*T_r + v*T_z) - kappa*Laplacian(T)
    
    return res_cont, res_mom_r, ...
```

---

### 3. Aspects of Symmetry in CZ Growth
Symmetry is the most important "prior knowledge" you can give the network. In the context of the paper, symmetry is handled in two ways:

#### A. Geometric Axisymmetry
By using $(r, z)$ as inputs instead of $(x, y, z)$, you are enforcing that the solution is identical at every angle $\theta$. This reduces a 3D problem into a 2D problem, drastically speeding up training.

#### B. Boundary Conditions at the Centerline ($r = 0$)
The physics requires specific "Symmetry BCs" at the axis of rotation ($r=0$). You must add these to your `bc_loss`:
1.  **Radial Velocity ($u$):** Must be $0$ at $r=0$.
2.  **Swirl Velocity ($w$):** Must be $0$ at $r=0$.
3.  **Temperature Gradient ($\partial T / \partial r$):** Must be $0$ at $r=0$ (Neumann boundary).
4.  **Axial Velocity Gradient ($\partial v / \partial r$):** Must be $0$ at $r=0$.

```python
# Sample points where r=0
X_axis = sample_axis(100) 
pred_axis = model(X_axis)
u_axis = pred_axis[:, 0:1]
# Derivative of T w.r.t r at the axis
dT_axis = grad(pred_axis[:, 3:4], X_axis)[:, 0:1]

loss_symmetry = torch.mean(u_axis**2) + torch.mean(dT_axis**2)
```

---

### 4. How to Train Effectively

The paper suggests a specific workflow for training SI-LB-PINNs:

#### Step 1: Data-Pretraining (The "Data" in SI-LB-PINN)
Don't start with the PDE residuals alone.
*   **Action:** If you have COMSOL data, train the network for 1,000 epochs using **only** the MSE loss between the network and the COMSOL data.
*   **Why:** This "warms up" the weights so the network doesn't have to find the complex flow structures (vortices) from scratch using only derivatives.

#### Step 2: Physics Refinement
*   **Action:** Introduce the PDE residuals and the Adaptive Weights (`log_vars`).
*   **Ratio:** The paper suggests using a high number of collocation points ($>5,000$) inside the melt, specifically clustered near the **crystal-melt interface** where gradients are steepest.

#### Step 3: Handling Rotation (The "Czochralski" Specifics)
The paper emphasizes the interaction between the crystal rotation ($+\omega$) and crucible rotation ($-\omega$). 
*   **The Crystal Boundary:** At the top surface ($z=H$), set $w = \Omega_{crystal} \cdot r$.
*   **The Crucible Boundary:** At the bottom and sides, set $w = \Omega_{crucible} \cdot r$.

#### Step 4: Nondimensionalization
Standard PINNs fail if your temperature is $1700$ K and velocity is $0.001$ m/s. 
*   **Action:** Scale all inputs $(r, z)$ to $[0, 1]$. 
*   **Action:** Scale all outputs $(u, v, T)$ so they are roughly in the range $[-1, 1]$. This is often done using the **Reynolds number (Re)** and **Grashof number (Gr)** formulations instead of raw SI units.

### Summary Checklist for your Notebook:
1.  [ ] **Add a 3rd velocity component ($w$)** for swirl/rotation.
2.  [ ] **Update `pde_residuals`** to use cylindrical coordinates.
3.  [ ] **Implement Symmetry BCs** at $r=0$.
4.  [ ] **Use a "Point Cloud"** that is denser near the crystal interface (Boundary Layer sampling).
5.  [ ] **Import CSV data** from COMSOL and add an `L_data` term to the loss to guide the SI-layers.
