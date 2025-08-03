"""
Integration tests for the Chaotic Pendulum Educational Project.

This package contains integration tests that verify the interaction
between different components of the chaotic pendulum system.
"""

__version__ = "0.1.0"
__author__ = "Jobet P. Casquejo"
__email__ = "jobetcasquejo221@gmail.com"

# Import common test utilities
from .conftest import *  # noqa: F403, F401

# Test markers for pytest
import pytest

pytest_plugins = [
    "tests.integration.conftest",
]

# Integration test categories
INTEGRATION_TESTS = {
    "simulation": "Tests for pendulum simulation functionality",
    "chaos_detection": "Tests for chaos detection algorithms",
    "data_processing": "Tests for data processing pipelines",
    "visualization": "Tests for plotting and visualization",
    "api": "Tests for API endpoints and interfaces",
    "database": "Tests for database operations",
    "performance": "Tests for performance and scalability",
}

def get_integration_test_categories():
    """Return available integration test categories."""
    return list(INTEGRATION_TESTS.keys())

def get_integration_test_description(category):
    """Get description for a specific integration test category."""
    return INTEGRATION_TESTS.get(category, "Unknown category") 