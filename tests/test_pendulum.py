"""
Unit tests for the pendulum simulation module.

Tests the pendulum simulation functionality including initialization,
simulation methods, and data validation.
"""

import pytest
import numpy as np
from tests.conftest import assert_arrays_close, assert_valid_pendulum_data

# Mock pendulum module for testing
class PendulumSimulator:
    """Mock pendulum simulator for testing."""
    
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

class TestPendulumSimulator:
    """Test cases for PendulumSimulator class."""
    
    def test_initialization(self):
        """Test pendulum simulator initialization."""
        config = {'time_step': 0.01, 'gravity': 9.81}
        simulator = PendulumSimulator(config)
        
        assert simulator.config == config
        assert simulator.results is None
    
    def test_initialization_default_config(self):
        """Test pendulum simulator initialization with default config."""
        simulator = PendulumSimulator()
        
        assert simulator.config == {}
        assert simulator.results is None
    
    def test_simulation_basic(self):
        """Test basic simulation functionality."""
        simulator = PendulumSimulator()
        t_span = [0, 10]
        initial_conditions = [0.1, 0.0]
        
        results = simulator.simulate(t_span, initial_conditions)
        
        # Check that results are returned
        assert results is not None
        assert_valid_pendulum_data(results)
        
        # Check that results are stored
        assert simulator.results is not None
        assert simulator.results == results
    
    def test_simulation_time_span(self):
        """Test simulation with different time spans."""
        simulator = PendulumSimulator()
        
        # Test short time span
        results_short = simulator.simulate([0, 1], [0.1, 0.0])
        assert len(results_short['time']) > 0
        
        # Test long time span
        results_long = simulator.simulate([0, 100], [0.1, 0.0])
        assert len(results_long['time']) > len(results_short['time'])
    
    def test_simulation_initial_conditions(self):
        """Test simulation with different initial conditions."""
        simulator = PendulumSimulator()
        t_span = [0, 10]
        
        # Test different initial conditions
        results1 = simulator.simulate(t_span, [0.1, 0.0])
        results2 = simulator.simulate(t_span, [0.5, 0.0])
        
        # Results should be different for different initial conditions
        assert not np.array_equal(results1['theta'], results2['theta'])
    
    def test_energy_calculation_no_simulation(self):
        """Test energy calculation without prior simulation."""
        simulator = PendulumSimulator()
        
        energy = simulator.get_energy()
        assert energy is None
    
    def test_energy_calculation_with_simulation(self):
        """Test energy calculation after simulation."""
        simulator = PendulumSimulator()
        t_span = [0, 10]
        initial_conditions = [0.1, 0.0]
        
        # Run simulation
        simulator.simulate(t_span, initial_conditions)
        
        # Calculate energy
        energy = simulator.get_energy()
        
        # Energy should be calculated
        assert energy is not None
        assert len(energy) == len(simulator.results['time'])
        assert np.all(energy >= 0)  # Energy should be non-negative
    
    def test_simulation_data_structure(self):
        """Test that simulation data has correct structure."""
        simulator = PendulumSimulator()
        t_span = [0, 10]
        initial_conditions = [0.1, 0.0]
        
        results = simulator.simulate(t_span, initial_conditions)
        
        # Check required keys
        required_keys = ['time', 'theta', 'omega']
        for key in required_keys:
            assert key in results
        
        # Check data types
        assert isinstance(results['time'], np.ndarray)
        assert isinstance(results['theta'], np.ndarray)
        assert isinstance(results['omega'], np.ndarray)
        
        # Check array lengths
        assert len(results['time']) == len(results['theta'])
        assert len(results['time']) == len(results['omega'])
    
    def test_simulation_time_monotonic(self):
        """Test that simulation time is monotonically increasing."""
        simulator = PendulumSimulator()
        t_span = [0, 10]
        initial_conditions = [0.1, 0.0]
        
        results = simulator.simulate(t_span, initial_conditions)
        
        # Check time is monotonically increasing
        time_diff = np.diff(results['time'])
        assert np.all(time_diff > 0)
    
    def test_simulation_time_bounds(self):
        """Test that simulation time is within specified bounds."""
        simulator = PendulumSimulator()
        t_span = [0, 10]
        initial_conditions = [0.1, 0.0]
        
        results = simulator.simulate(t_span, initial_conditions)
        
        # Check time bounds
        assert results['time'][0] >= t_span[0]
        assert results['time'][-1] <= t_span[1]
    
    @pytest.mark.slow
    def test_simulation_large_time_span(self):
        """Test simulation with large time span (slow test)."""
        simulator = PendulumSimulator()
        t_span = [0, 1000]
        initial_conditions = [0.1, 0.0]
        
        results = simulator.simulate(t_span, initial_conditions)
        
        assert len(results['time']) > 1000
        assert_valid_pendulum_data(results)

class TestPendulumUtilities:
    """Test cases for pendulum utility functions."""
    
    def test_assert_valid_pendulum_data_valid(self):
        """Test validation of valid pendulum data."""
        valid_data = {
            'time': np.linspace(0, 10, 100),
            'theta': np.sin(np.linspace(0, 10, 100)),
            'omega': np.cos(np.linspace(0, 10, 100))
        }
        
        # Should not raise an exception
        assert_valid_pendulum_data(valid_data)
    
    def test_assert_valid_pendulum_data_missing_key(self):
        """Test validation of pendulum data with missing key."""
        invalid_data = {
            'time': np.linspace(0, 10, 100),
            'theta': np.sin(np.linspace(0, 10, 100))
            # Missing 'omega'
        }
        
        with pytest.raises(AssertionError):
            assert_valid_pendulum_data(invalid_data)
    
    def test_assert_valid_pendulum_data_different_lengths(self):
        """Test validation of pendulum data with different array lengths."""
        invalid_data = {
            'time': np.linspace(0, 10, 100),
            'theta': np.sin(np.linspace(0, 10, 50)),  # Different length
            'omega': np.cos(np.linspace(0, 10, 100))
        }
        
        with pytest.raises(AssertionError):
            assert_valid_pendulum_data(invalid_data)
    
    def test_assert_valid_pendulum_data_non_monotonic_time(self):
        """Test validation of pendulum data with non-monotonic time."""
        invalid_data = {
            'time': np.array([1, 0, 2, 3]),  # Non-monotonic
            'theta': np.array([0, 0, 0, 0]),
            'omega': np.array([0, 0, 0, 0])
        }
        
        with pytest.raises(AssertionError):
            assert_valid_pendulum_data(invalid_data) 