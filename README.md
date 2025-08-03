# Chaotic Pendulum Educational Project

A comprehensive educational project for studying chaotic dynamics in driven pendulum systems. This project provides tools for simulating, analyzing, and visualizing chaotic behavior in pendulum systems, making it ideal for physics education and research.

## 🎯 Project Overview

This project explores the fascinating world of chaotic dynamics through the lens of a driven pendulum system. The pendulum exhibits chaotic behavior under certain conditions, making it an excellent case study for understanding nonlinear dynamics, chaos theory, and complex systems.

### Key Features

- **Chaotic Pendulum Simulation**: Simulate driven pendulum systems with configurable parameters
- **Chaos Detection**: Implement Lyapunov exponent analysis to detect chaotic behavior
- **Data Analysis**: Comprehensive tools for analyzing simulation results
- **Educational Content**: Quizzes and interactive learning materials
- **Visualization**: Rich plotting and animation capabilities
- **Batch Processing**: Run multiple simulations with different parameters

## 📁 Project Structure

```
chaotic_pendulum_edu/
├── animations/           # Generated animations and visualizations
├── data/                # Simulation data and results
│   ├── features/        # Extracted features from simulations
│   ├── labeled/         # Labeled datasets for machine learning
│   ├── raw/            # Raw simulation data
│   └── sample_plots/   # Sample visualization outputs
├── models/              # Trained machine learning models
├── notebooks/           # Jupyter notebooks for analysis
├── quizzes/            # Educational quizzes and assessments
├── reports/            # Generated reports and documentation
│   └── figures/        # Report figures and plots
├── src/                # Source code modules
├── student_submissions/ # Student work and submissions
├── tests/              # Unit tests and validation
├── environment.yml     # Conda environment configuration
├── pendulum_config.json # Main configuration file
└── requirements.txt    # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- Conda (recommended) or pip

### Installation

#### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/jobet1130/chaotic_pendulum_edu.git
cd chaotic_pendulum_edu

# Create and activate the conda environment
conda env create -f environment.yml
conda activate pendulum-chaos-env
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/jobet1130/chaotic_pendulum_edu.git
cd chaotic_pendulum_edu

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

The project uses `pendulum_config.json` for configuration. Key parameters include:

- **Physical Parameters**: Gravity, pendulum length, mass, damping
- **Initial Conditions**: Initial angle and velocity
- **Driving Force**: Amplitude and frequency of external driving
- **Simulation Settings**: Time step, total simulation time
- **Chaos Detection**: Lyapunov exponent analysis parameters
- **Integration**: RK4 method for numerical integration

## 🔧 Usage

### Basic Simulation

```python
import json
from src.pendulum_simulator import PendulumSimulator

# Load configuration
with open('pendulum_config.json', 'r') as f:
    config = json.load(f)

# Create simulator
simulator = PendulumSimulator(config)

# Run simulation
results = simulator.run_simulation()

# Analyze results
simulator.analyze_chaos(results)
```

### Batch Simulations

```python
# Run multiple simulations with different parameters
batch_results = simulator.run_batch_simulations()
```

### Visualization

```python
# Generate phase space plots
simulator.plot_phase_space(results)

# Create animations
simulator.create_animation(results)
```

## 📊 Key Features

### 1. Chaotic Dynamics Analysis
- **Lyapunov Exponents**: Detect chaotic behavior using Rosenstein's method
- **Phase Space Analysis**: Visualize attractors and trajectories
- **Bifurcation Diagrams**: Study parameter dependence of system behavior

### 2. Educational Tools
- **Interactive Notebooks**: Step-by-step tutorials and examples
- **Quizzes**: Assessment tools for learning verification
- **Visualizations**: Rich plotting capabilities for understanding concepts

### 3. Data Management
- **Structured Data Storage**: Organized data hierarchy
- **Feature Extraction**: Automated feature computation
- **Export Formats**: Multiple output formats (CSV, JSON, etc.)

## 🧪 Educational Applications

This project is designed for:

- **Physics Education**: Understanding nonlinear dynamics and chaos
- **Mathematics**: Exploring differential equations and numerical methods
- **Computer Science**: Learning scientific computing and data analysis
- **Research**: Investigating complex systems and emergent behavior

## 📚 Learning Objectives

By working with this project, students will learn:

1. **Chaos Theory**: Understanding deterministic chaos and sensitivity to initial conditions
2. **Nonlinear Dynamics**: Exploring bifurcations, attractors, and phase space
3. **Numerical Methods**: Implementing and using RK4 integration
4. **Data Analysis**: Processing and visualizing complex time series data
5. **Scientific Computing**: Using Python for scientific simulations

## 🔬 Research Applications

The project supports research in:

- **Dynamical Systems**: Study of complex behavior in simple systems
- **Chaos Detection**: Development of algorithms for chaos identification
- **Machine Learning**: Feature extraction and pattern recognition in chaotic data
- **Physics Education**: Development of interactive learning materials

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- Code improvements and bug fixes
- New educational content and examples
- Documentation enhancements
- Feature requests and suggestions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🙏 Acknowledgments

- Physics educators and researchers who inspired this project
- Open-source community for the scientific Python ecosystem
- Students and educators who provide feedback and suggestions

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please contact:

**Jobet P. Casquejo**  
📧 [jobetcasquejo221@gmail.com](mailto:jobetcasquejo221@gmail.com)

---

**Note**: This project is designed for educational purposes. For research applications, please ensure proper validation and peer review of results.
