"""
Test suite for the Chaotic Pendulum Educational Project.

This package contains unit tests, integration tests, and other test utilities
for the chaotic pendulum educational project.
"""

__version__ = "0.1.0"
__author__ = "Jobet P. Casquejo"
__email__ = "jobetcasquejo221@gmail.com"

# Test configuration
import pytest

# Test markers
pytest_plugins = [
    "tests.conftest",
]

# Test categories
TEST_CATEGORIES = {
    "unit": "Unit tests for individual functions and classes",
    "integration": "Integration tests for component interactions",
    "simulation": "Tests for pendulum simulation functionality",
    "chaos": "Tests for chaos detection algorithms",
    "visualization": "Tests for plotting and visualization",
    "performance": "Tests for performance and scalability",
    "slow": "Tests that take longer to run",
}

def get_test_categories():
    """Return available test categories."""
    return list(TEST_CATEGORIES.keys())

def get_test_description(category):
    """Get description for a specific test category."""
    return TEST_CATEGORIES.get(category, "Unknown category") 