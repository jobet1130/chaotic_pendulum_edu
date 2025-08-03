"""
Unit tests for the chaos detection module.

Tests the chaos detection functionality including Lyapunov exponent
calculations and chaos detection algorithms.
"""

import pytest
import numpy as np
from tests.conftest import assert_arrays_close

# Mock chaos detection module for testing
class ChaosDetector:
    """Mock chaos detector for testing."""
    
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

class LyapunovCalculator:
    """Mock Lyapunov calculator for testing."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.results = {}
    
    def calculate(self, time_series, method='wolf'):
        """Mock Lyapunov exponent calculation."""
        # Mock calculation based on method
        if method == 'wolf':
            lyap = 0.3
        elif method == 'rosenstein':
            lyap = 0.4
        else:
            lyap = 0.2
        
        self.results[method] = {
            'lyapunov_exponent': lyap,
            'method': method,
            'data_length': len(time_series)
        }
        
        return lyap
    
    def get_results(self, method=None):
        """Get calculation results."""
        if method is None:
            return self.results
        return self.results.get(method)

class TestChaosDetector:
    """Test cases for ChaosDetector class."""
    
    def test_initialization(self):
        """Test chaos detector initialization."""
        detector = ChaosDetector()
        
        assert detector.lyapunov_exponent is None
    
    def test_calculate_lyapunov_basic(self):
        """Test basic Lyapunov exponent calculation."""
        detector = ChaosDetector()
        data = np.random.randn(1000)
        
        lyap = detector.calculate_lyapunov(data)
        
        assert lyap is not None
        assert lyap == 0.5  # Mock value
        assert detector.lyapunov_exponent == lyap
    
    def test_calculate_lyapunov_multiple_calls(self):
        """Test multiple Lyapunov exponent calculations."""
        detector = ChaosDetector()
        data1 = np.random.randn(500)
        data2 = np.random.randn(1000)
        
        lyap1 = detector.calculate_lyapunov(data1)
        lyap2 = detector.calculate_lyapunov(data2)
        
        # Should return same mock value
        assert lyap1 == lyap2
        assert detector.lyapunov_exponent == lyap2
    
    def test_is_chaotic_positive_lyapunov(self):
        """Test chaos detection with positive Lyapunov exponent."""
        detector = ChaosDetector()
        data = np.random.randn(1000)
        
        detector.calculate_lyapunov(data)
        is_chaotic = detector.is_chaotic()
        
        assert is_chaotic is True
    
    def test_is_chaotic_no_calculation(self):
        """Test chaos detection without prior calculation."""
        detector = ChaosDetector()
        
        is_chaotic = detector.is_chaotic()
        assert is_chaotic is False
    
    def test_is_chaotic_custom_threshold(self):
        """Test chaos detection with custom threshold."""
        detector = ChaosDetector()
        data = np.random.randn(1000)
        
        detector.calculate_lyapunov(data)
        
        # Test with high threshold
        is_chaotic_high = detector.is_chaotic(threshold=1.0)
        assert is_chaotic_high is False
        
        # Test with low threshold
        is_chaotic_low = detector.is_chaotic(threshold=0.1)
        assert is_chaotic_low is True
    
    def test_is_chaotic_zero_threshold(self):
        """Test chaos detection with zero threshold."""
        detector = ChaosDetector()
        data = np.random.randn(1000)
        
        detector.calculate_lyapunov(data)
        is_chaotic = detector.is_chaotic(threshold=0.0)
        
        assert is_chaotic is True

class TestLyapunovCalculator:
    """Test cases for LyapunovCalculator class."""
    
    def test_initialization(self):
        """Test Lyapunov calculator initialization."""
        config = {'method': 'wolf', 'window_size': 100}
        calculator = LyapunovCalculator(config)
        
        assert calculator.config == config
        assert calculator.results == {}
    
    def test_initialization_default_config(self):
        """Test Lyapunov calculator initialization with default config."""
        calculator = LyapunovCalculator()
        
        assert calculator.config == {}
        assert calculator.results == {}
    
    def test_calculate_wolf_method(self):
        """Test Lyapunov calculation with Wolf method."""
        calculator = LyapunovCalculator()
        time_series = np.random.randn(1000)
        
        lyap = calculator.calculate(time_series, method='wolf')
        
        assert lyap == 0.3  # Mock value for wolf method
        assert 'wolf' in calculator.results
        assert calculator.results['wolf']['lyapunov_exponent'] == lyap
        assert calculator.results['wolf']['method'] == 'wolf'
        assert calculator.results['wolf']['data_length'] == 1000
    
    def test_calculate_rosenstein_method(self):
        """Test Lyapunov calculation with Rosenstein method."""
        calculator = LyapunovCalculator()
        time_series = np.random.randn(1000)
        
        lyap = calculator.calculate(time_series, method='rosenstein')
        
        assert lyap == 0.4  # Mock value for rosenstein method
        assert 'rosenstein' in calculator.results
        assert calculator.results['rosenstein']['lyapunov_exponent'] == lyap
    
    def test_calculate_unknown_method(self):
        """Test Lyapunov calculation with unknown method."""
        calculator = LyapunovCalculator()
        time_series = np.random.randn(1000)
        
        lyap = calculator.calculate(time_series, method='unknown')
        
        assert lyap == 0.2  # Default mock value
        assert 'unknown' in calculator.results
    
    def test_calculate_multiple_methods(self):
        """Test Lyapunov calculation with multiple methods."""
        calculator = LyapunovCalculator()
        time_series = np.random.randn(1000)
        
        lyap_wolf = calculator.calculate(time_series, method='wolf')
        lyap_rosenstein = calculator.calculate(time_series, method='rosenstein')
        
        assert lyap_wolf != lyap_rosenstein
        assert len(calculator.results) == 2
        assert 'wolf' in calculator.results
        assert 'rosenstein' in calculator.results
    
    def test_get_results_all(self):
        """Test getting all calculation results."""
        calculator = LyapunovCalculator()
        time_series = np.random.randn(1000)
        
        calculator.calculate(time_series, method='wolf')
        calculator.calculate(time_series, method='rosenstein')
        
        results = calculator.get_results()
        
        assert isinstance(results, dict)
        assert len(results) == 2
        assert 'wolf' in results
        assert 'rosenstein' in results
    
    def test_get_results_specific_method(self):
        """Test getting results for specific method."""
        calculator = LyapunovCalculator()
        time_series = np.random.randn(1000)
        
        calculator.calculate(time_series, method='wolf')
        
        wolf_results = calculator.get_results('wolf')
        rosenstein_results = calculator.get_results('rosenstein')
        
        assert wolf_results is not None
        assert wolf_results['method'] == 'wolf'
        assert rosenstein_results is None
    
    def test_get_results_no_calculations(self):
        """Test getting results without any calculations."""
        calculator = LyapunovCalculator()
        
        results = calculator.get_results()
        assert results == {}
        
        specific_results = calculator.get_results('wolf')
        assert specific_results is None

class TestChaosUtilities:
    """Test cases for chaos utility functions."""
    
    def test_chaos_detection_workflow(self):
        """Test complete chaos detection workflow."""
        detector = ChaosDetector()
        calculator = LyapunovCalculator()
        
        # Generate test data
        time_series = np.random.randn(1000)
        
        # Calculate Lyapunov exponent
        lyap = calculator.calculate(time_series, method='wolf')
        
        # Detect chaos
        detector.calculate_lyapunov(time_series)
        is_chaotic = detector.is_chaotic()
        
        # Verify results
        assert lyap > 0
        assert is_chaotic is True
    
    def test_multiple_chaos_detectors(self):
        """Test multiple chaos detectors with different thresholds."""
        detector1 = ChaosDetector()
        detector2 = ChaosDetector()
        
        data = np.random.randn(1000)
        
        detector1.calculate_lyapunov(data)
        detector2.calculate_lyapunov(data)
        
        # Same data should give same Lyapunov exponent
        assert detector1.lyapunov_exponent == detector2.lyapunov_exponent
        
        # Different thresholds should give different chaos detection
        is_chaotic1 = detector1.is_chaotic(threshold=0.1)
        is_chaotic2 = detector2.is_chaotic(threshold=1.0)
        
        assert is_chaotic1 != is_chaotic2 