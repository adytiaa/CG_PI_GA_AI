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

