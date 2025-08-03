"""
Common test fixtures and utilities for the Chaotic Pendulum Educational Project.

This module provides shared fixtures, utilities, and configuration for all tests.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Test data and configuration
@pytest.fixture
def sample_pendulum_data():
    """Provide sample pendulum simulation data for testing."""
    t = np.linspace(0, 10, 1000)
    theta = np.sin(t) + 0.1 * np.random.randn(len(t))
    omega = np.cos(t) + 0.1 * np.random.randn(len(t))
    
    return {
        'time': t,
        'theta': theta,
        'omega': omega,
        'energy': 0.5 * omega**2 + np.cos(theta)
    }

@pytest.fixture
def sample_chaos_data():
    """Provide sample chaos detection data for testing."""
    # Logistic map data for chaos testing
    r = 3.9
    x = np.zeros(1000)
    x[0] = 0.5
    for i in range(1, len(x)):
        x[i] = r * x[i-1] * (1 - x[i-1])
    
    return {
        'time': np.arange(len(x)),
        'values': x,
        'parameter': r
    }

@pytest.fixture
def temp_data_dir():
    """Provide a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_config():
    """Provide sample configuration for testing."""
    return {
        'simulation': {
            'time_step': 0.01,
            'total_time': 10.0,
            'initial_conditions': [0.1, 0.0],
            'parameters': {
                'gravity': 9.81,
                'length': 1.0,
                'damping': 0.1,
                'forcing_amplitude': 0.5,
                'forcing_frequency': 1.0
            }
        },
        'analysis': {
            'lyapunov_samples': 1000,
            'bifurcation_points': 100,
            'chaos_threshold': 0.01
        },
        'visualization': {
            'figure_size': [10, 8],
            'dpi': 100,
            'save_format': 'png'
        }
    }

@pytest.fixture
def mock_pendulum_simulator():
    """Provide a mock pendulum simulator for testing."""
    class MockPendulumSimulator:
        def __init__(self, config=None):
            self.config = config or {}
            self.results = None
        
        def simulate(self, t_span, initial_conditions):
            """Mock simulation method."""
            t = np.linspace(t_span[0], t_span[1], 1000)
            theta = np.sin(t) + 0.1 * np.random.randn(len(t))
            omega = np.cos(t) + 0.1 * np.random.randn(len(t))
            
            self.results = {
                'time': t,
                'theta': theta,
                'omega': omega
            }
            return self.results
        
        def get_energy(self):
            """Mock energy calculation."""
            if self.results is None:
                return None
            return 0.5 * self.results['omega']**2 + np.cos(self.results['theta'])
    
    return MockPendulumSimulator

@pytest.fixture
def mock_chaos_detector():
    """Provide a mock chaos detector for testing."""
    class MockChaosDetector:
        def __init__(self):
            self.lyapunov_exponent = None
        
        def calculate_lyapunov(self, data):
            """Mock Lyapunov exponent calculation."""
            # Return a mock positive Lyapunov exponent (chaotic)
            self.lyapunov_exponent = 0.5
            return self.lyapunov_exponent
        
        def is_chaotic(self, threshold=0.01):
            """Mock chaos detection."""
            if self.lyapunov_exponent is None:
                return False
            return self.lyapunov_exponent > threshold
    
    return MockChaosDetector

@pytest.fixture
def mock_plot_manager():
    """Provide a mock plot manager for testing."""
    class MockPlotManager:
        def __init__(self):
            self.figures = []
        
        def create_phase_plot(self, data):
            """Mock phase plot creation."""
            fig = {'type': 'phase_plot', 'data': data}
            self.figures.append(fig)
            return fig
        
        def create_time_series(self, data):
            """Mock time series plot creation."""
            fig = {'type': 'time_series', 'data': data}
            self.figures.append(fig)
            return fig
        
        def save_figure(self, fig, filename):
            """Mock figure saving."""
            return f"Saved {fig['type']} to {filename}"
    
    return MockPlotManager

# Test utilities
def assert_arrays_close(actual, expected, rtol=1e-5, atol=1e-8):
    """Assert that two arrays are close within tolerance."""
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

def assert_dict_contains(dict_obj, required_keys):
    """Assert that a dictionary contains all required keys."""
    for key in required_keys:
        assert key in dict_obj, f"Missing required key: {key}"

def assert_valid_pendulum_data(data):
    """Assert that pendulum data has valid structure and values."""
    required_keys = ['time', 'theta', 'omega']
    assert_dict_contains(data, required_keys)
    
    # Check that arrays have same length
    lengths = [len(data[key]) for key in required_keys]
    assert len(set(lengths)) == 1, "All arrays must have same length"
    
    # Check that time is monotonically increasing
    assert np.all(np.diff(data['time']) > 0), "Time must be monotonically increasing"

def create_test_file(filename, content=""):
    """Create a test file with given content."""
    with open(filename, 'w') as f:
        f.write(content)
    return filename

def cleanup_test_file(filename):
    """Clean up a test file."""
    if os.path.exists(filename):
        os.remove(filename)

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual functions and classes"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "simulation: Tests for pendulum simulation functionality"
    )
    config.addinivalue_line(
        "markers", "chaos: Tests for chaos detection algorithms"
    )
    config.addinivalue_line(
        "markers", "visualization: Tests for plotting and visualization"
    )
    config.addinivalue_line(
        "markers", "performance: Tests for performance and scalability"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    ) 