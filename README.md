# Physics-Based AI modes for acceerating and increasing the Efficiency of CFD for Silicon Single Crystal Growth Simulations


A modular framework implementing variants of Physics-Informed Neural Networks (PINNs) for thermal-fluid coupling which includes Navier-Stokes equations CFD for crystal growth simulations.

## Features

- **Modular Design**: Easily extendable framework for PINNs with clean separation of concerns
- **Gradient Normalization**: Improved training stability using advanced gradient normalization techniques
- **Multiple PDE Support**:
  - Navier-Stokes Equations for Crystal Growth (2D)
  - Navier-Stokes Equations for Crystal Growth (3D)
- **Neural Network **:
  - Physics-Inspired NNs
  - PI-GANO (GEOMETRY AWARE)
  - GAOT
- **Visualization Tools**: Comprehensive plotting utilities for solution visualization

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- SciPy
- scikit-learn

## Installation

```bash
# Clone the repository
git clone repo url
cd dir/

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The framework can be used through the command line interface:

```bash
python main.py --equation [heat|schrodinger|kdv|crystal] [options]
```

### Command Line Arguments

- `--equation`: Type of equation to solve (heat, schrodinger, kdv, crystal)
- `--use_gn`: Use gradient normalization for improved training
- `--network_type`: Neural network type (mlp or siren)
- `--hidden_layers`: Number of hidden layers in the neural network
- `--neurons`: Number of neurons per hidden layer
- `--learning_rate`: Learning rate for optimization
- `--epochs`: Number of training epochs
- `--device`: Device to use (cpu or cuda)
- `--output_dir`: Directory to save results

For Navier-Stokes crystal growth simulation:
- `--viscosity`: Kinematic viscosity coefficient
- `--thermal_diffusivity`: Thermal diffusivity
- `--density`: Fluid density
- `--plot_resolution`: Resolution for visualization plots

### Examples


#### Crystal Growth Simulation using Navier-Stokes
```bash
python main.py --equation crystal --use_gn --network_type siren --hidden_layers 5 --neurons 100 --viscosity 0.01 --thermal_diffusivity 0.005 --epochs 5000
```

## Project Structure



## Future Work

- Add support for higher-dimensional and Thermal Coupling Equations
- Add more visualization options
- Extend crystal growth modeling capabilities
- Implement 3D Navier-Stokes equations


## Acknowledgments

- Based on the PINN framework developed by Raissi et al.
- Gradient normalization techniques inspired by Wang et al.
- SIREN implementation based on the paper by Sitzmann et al.

# Crystal Growth Simulation

This project simulates crystal growth using the Navier-Stokes, thermal coupling, and variants of Physics-Informed Geometry Aware AI mode
## Setup

The project uses a virtual environment named "gradient" for dependency management.

### Activating the Virtual Environment

```bash
# On macOS/Linux
source gradient/bin/activate

# On Windows
gradient\Scripts\activate
```

### Required Dependencies

After activating the virtual environment, ensure you have the following dependencies:

```bash
pip install numpy matplotlib torch
```

## Running the Simulation

### Simple Animation

To generate a series of frames showing the crystal growth:

```bash
python examples/simple_animation.py --frames 50
```

Options:
- `--frames`: Number of frames to generate (default: 50)
- `--output-dir`: Custom directory to save frames (optional)

After running the script, the frames will be saved in `results/simple_animation/frames/` and an HTML slideshow will be created at `results/simple_animation/slideshow.html`.

### Advanced Simulation with PINNs

For the full physics-informed neural network simulation:

```bash
python examples/crystal_growth_video.py
```

Options:
- `--frames`: Number of frames in the video
- `--duration`: Duration of the video in seconds
- `--fps`: Frames per second
- `--model`: Path to saved model (optional)
- `--train`: Force training a new model

## File Structure

- `equations/`: Contains the physical equations including Navier-Stokes
- `models/`: Neural network architectures (MLP, SIREN, GNPINN)
- `utils/`: Utility functions for training and visualization
- `examples/`: Example scripts
  - `simple_animation.py`: Creates frame-by-frame animation
  - `crystal_growth_video.py`: Full PINN-based simulation

## Visualization

The simulation creates both individual frames and an HTML slideshow for viewing the results. Open the slideshow in a web browser to see the animation with playback controls

## License

This project is licensed under the MIT License - see the LICENSE file for details.
