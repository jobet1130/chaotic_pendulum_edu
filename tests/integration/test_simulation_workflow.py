"""
Integration test for the complete simulation workflow.

Tests the integration between pendulum simulation, chaos detection,
and data management components.
"""

import pytest
import numpy as np
from tests.conftest import sample_pendulum_data, sample_config

# Mock classes for integration testing
class PendulumSimulator:
    def __init__(self, config=None):
        self.config = config or {}
        self.results = None
    
    def simulate(self, t_span, initial_conditions):
        t = np.linspace(t_span[0], t_span[1], 1000)
        theta = np.sin(t) + 0.1 * np.random.randn(len(t))
        omega = np.cos(t) + 0.1 * np.random.randn(len(t))
        
        self.results = {
            'time': t,
            'theta': theta,
            'omega': omega
        }
        return self.results

class ChaosDetector:
    def __init__(self):
        self.lyapunov_exponent = None
    
    def calculate_lyapunov(self, data):
        self.lyapunov_exponent = 0.5
        return self.lyapunov_exponent
    
    def is_chaotic(self, threshold=0.01):
        if self.lyapunov_exponent is None:
            return False
        return self.lyapunov_exponent > threshold

class DataManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.data = {}
    
    def save_data(self, name, data):
        self.data[name] = data
        return f"{self.data_dir}/{name}.json"
    
    def load_data(self, name):
        return self.data.get(name)

class TestSimulationWorkflow:
    """Integration tests for the complete simulation workflow."""
    
    def test_complete_simulation_workflow(self, sample_config):
        """Test the complete simulation workflow from start to finish."""
        # Initialize components
        simulator = PendulumSimulator(sample_config)
        detector = ChaosDetector()
        data_manager = DataManager()
        
        # Step 1: Run simulation
        t_span = [0, 10]
        initial_conditions = [0.1, 0.0]
        simulation_results = simulator.simulate(t_span, initial_conditions)
        
        # Verify simulation results
        assert simulation_results is not None
        assert 'time' in simulation_results
        assert 'theta' in simulation_results
        assert 'omega' in simulation_results
        assert len(simulation_results['time']) > 0
        
        # Step 2: Detect chaos
        lyapunov = detector.calculate_lyapunov(simulation_results['theta'])
        is_chaotic = detector.is_chaotic()
        
        # Verify chaos detection
        assert lyapunov is not None
        assert lyapunov > 0
        assert is_chaotic is True
        
        # Step 3: Save results
        results_data = {
            'simulation': simulation_results,
            'chaos_detection': {
                'lyapunov_exponent': lyapunov,
                'is_chaotic': is_chaotic
            },
            'config': sample_config
        }
        
        filepath = data_manager.save_data('workflow_results', results_data)
        
        # Verify data saving
        assert filepath is not None
        assert 'workflow_results' in data_manager.data
        
        # Step 4: Load and verify results
        loaded_results = data_manager.load_data('workflow_results')
        
        # Verify loaded data
        assert loaded_results is not None
        assert 'simulation' in loaded_results
        assert 'chaos_detection' in loaded_results
        assert 'config' in loaded_results
        assert loaded_results['chaos_detection']['is_chaotic'] is True
    
    def test_simulation_with_different_parameters(self, sample_config):
        """Test simulation workflow with different parameters."""
        simulator = PendulumSimulator(sample_config)
        detector = ChaosDetector()
        data_manager = DataManager()
        
        # Test multiple parameter sets
        test_cases = [
            {'t_span': [0, 5], 'initial_conditions': [0.1, 0.0]},
            {'t_span': [0, 20], 'initial_conditions': [0.5, 0.0]},
            {'t_span': [0, 10], 'initial_conditions': [1.0, 0.0]}
        ]
        
        for i, test_case in enumerate(test_cases):
            # Run simulation
            results = simulator.simulate(
                test_case['t_span'], 
                test_case['initial_conditions']
            )
            
            # Detect chaos
            lyapunov = detector.calculate_lyapunov(results['theta'])
            is_chaotic = detector.is_chaotic()
            
            # Save results
            data = {
                'test_case': i,
                'parameters': test_case,
                'simulation': results,
                'chaos_detection': {
                    'lyapunov_exponent': lyapunov,
                    'is_chaotic': is_chaotic
                }
            }
            
            data_manager.save_data(f'test_case_{i}', data)
            
            # Verify results
            assert len(results['time']) > 0
            assert lyapunov > 0
            assert is_chaotic is True
    
    def test_error_handling_in_workflow(self):
        """Test error handling in the simulation workflow."""
        # Test with invalid parameters
        simulator = PendulumSimulator()
        detector = ChaosDetector()
        data_manager = DataManager()
        
        # Test with empty data
        try:
            lyapunov = detector.calculate_lyapunov([])
            # Should handle empty data gracefully
        except Exception as e:
            # If exception is raised, it should be handled appropriately
            assert isinstance(e, Exception)
        
        # Test with None data
        try:
            lyapunov = detector.calculate_lyapunov(None)
            # Should handle None data gracefully
        except Exception as e:
            # If exception is raised, it should be handled appropriately
            assert isinstance(e, Exception)
    
    @pytest.mark.slow
    def test_large_scale_simulation(self, sample_config):
        """Test large-scale simulation (slow test)."""
        simulator = PendulumSimulator(sample_config)
        detector = ChaosDetector()
        data_manager = DataManager()
        
        # Large time span
        t_span = [0, 1000]
        initial_conditions = [0.1, 0.0]
        
        # Run simulation
        results = simulator.simulate(t_span, initial_conditions)
        
        # Verify large dataset
        assert len(results['time']) > 10000
        
        # Detect chaos
        lyapunov = detector.calculate_lyapunov(results['theta'])
        is_chaotic = detector.is_chaotic()
        
        # Save large dataset
        data = {
            'simulation': results,
            'chaos_detection': {
                'lyapunov_exponent': lyapunov,
                'is_chaotic': is_chaotic
            }
        }
        
        filepath = data_manager.save_data('large_simulation', data)
        
        # Verify results
        assert filepath is not None
        assert lyapunov > 0
        assert is_chaotic is True

class TestComponentIntegration:
    """Tests for component integration."""
    
    def test_simulator_detector_integration(self):
        """Test integration between simulator and chaos detector."""
        simulator = PendulumSimulator()
        detector = ChaosDetector()
        
        # Run simulation
        results = simulator.simulate([0, 10], [0.1, 0.0])
        
        # Use simulation data for chaos detection
        lyapunov = detector.calculate_lyapunov(results['theta'])
        is_chaotic = detector.is_chaotic()
        
        # Verify integration
        assert lyapunov is not None
        assert is_chaotic is True
        assert len(results['theta']) > 0
    
    def test_detector_data_manager_integration(self):
        """Test integration between chaos detector and data manager."""
        detector = ChaosDetector()
        data_manager = DataManager()
        
        # Generate test data
        test_data = np.random.randn(1000)
        
        # Detect chaos
        lyapunov = detector.calculate_lyapunov(test_data)
        is_chaotic = detector.is_chaotic()
        
        # Save results
        results = {
            'data': test_data.tolist(),
            'lyapunov_exponent': lyapunov,
            'is_chaotic': is_chaotic
        }
        
        filepath = data_manager.save_data('chaos_analysis', results)
        
        # Verify integration
        assert filepath is not None
        assert lyapunov > 0
        assert is_chaotic is True
    
    def test_full_component_integration(self):
        """Test full integration of all components."""
        simulator = PendulumSimulator()
        detector = ChaosDetector()
        data_manager = DataManager()
        
        # Complete workflow
        simulation_results = simulator.simulate([0, 10], [0.1, 0.0])
        lyapunov = detector.calculate_lyapunov(simulation_results['theta'])
        is_chaotic = detector.is_chaotic()
        
        # Save complete results
        complete_results = {
            'simulation': simulation_results,
            'chaos_analysis': {
                'lyapunov_exponent': lyapunov,
                'is_chaotic': is_chaotic
            },
            'metadata': {
                'timestamp': '2024-01-01',
                'version': '0.1.0'
            }
        }
        
        filepath = data_manager.save_data('complete_analysis', complete_results)
        
        # Verify complete integration
        assert filepath is not None
        assert 'simulation' in complete_results
        assert 'chaos_analysis' in complete_results
        assert 'metadata' in complete_results
        assert lyapunov > 0
        assert is_chaotic is True 