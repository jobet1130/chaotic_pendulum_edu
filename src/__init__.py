"""
Chaotic Pendulum Educational Project

A comprehensive educational project for studying chaotic dynamics 
in driven pendulum systems through simulation, analysis, and visualization.
"""

__version__ = "0.1.0"
__author__ = "Jobet P. Casquejo"
__email__ = "jobetcasquejo221@gmail.com"
__description__ = "Educational project for chaotic pendulum dynamics"
__url__ = "https://github.com/jobet1130/chaotic_pendulum_edu"

# Core modules - import conditionally to handle missing modules
try:
    from . import pendulum
except ImportError:
    pendulum = None

try:
    from . import chaos
except ImportError:
    chaos = None

try:
    from . import visualization
except ImportError:
    visualization = None

try:
    from . import analysis
except ImportError:
    analysis = None

try:
    from . import utils
except ImportError:
    utils = None

# Main classes and functions for easy import - import conditionally
try:
    from .pendulum import PendulumSimulator
except ImportError:
    PendulumSimulator = None

try:
    from .chaos import ChaosDetector, LyapunovCalculator
except ImportError:
    ChaosDetector = None
    LyapunovCalculator = None

try:
    from .visualization import PlotManager, AnimationBuilder
except ImportError:
    PlotManager = None
    AnimationBuilder = None

try:
    from .analysis import DataAnalyzer, BifurcationAnalyzer
except ImportError:
    DataAnalyzer = None
    BifurcationAnalyzer = None

try:
    from .utils import ConfigManager, DataManager
except ImportError:
    ConfigManager = None
    DataManager = None

# Version info
__all__ = [
    # Core modules
    "pendulum",
    "chaos", 
    "visualization",
    "analysis",
    "utils",
    
    # Main classes
    "PendulumSimulator",
    "ChaosDetector",
    "LyapunovCalculator", 
    "PlotManager",
    "AnimationBuilder",
    "DataAnalyzer",
    "BifurcationAnalyzer",
    "ConfigManager",
    "DataManager",
    
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__url__",
]

# Package metadata
PACKAGE_INFO = {
    "name": "chaotic-pendulum-edu",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "email": __email__,
    "url": __url__,
    "keywords": [
        "chaos", "pendulum", "dynamics", "education", "physics",
        "nonlinear", "simulation", "lyapunov", "bifurcation"
    ],
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
    ]
}

def get_package_info():
    """Return package information dictionary."""
    return PACKAGE_INFO.copy()

def get_version():
    """Return the package version."""
    return __version__

def get_author():
    """Return the package author."""
    return __author__

def get_description():
    """Return the package description."""
    return __description__

# Initialize package
def initialize_package():
    """Initialize the chaotic pendulum educational package."""
    print(f"Chaotic Pendulum Educational Project v{__version__}")
    print(f"Author: {__author__} ({__email__})")
    print(f"Description: {__description__}")
    print(f"URL: {__url__}")
    
    # Check module availability
    modules_status = {
        "pendulum": pendulum is not None,
        "chaos": chaos is not None,
        "visualization": visualization is not None,
        "analysis": analysis is not None,
        "utils": utils is not None
    }
    
    print("\nModule Status:")
    for module, available in modules_status.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {module}: {status}")
    
    print("\nPackage initialized successfully!")

# Auto-initialize when imported
if __name__ == "__main__":
    initialize_package()
