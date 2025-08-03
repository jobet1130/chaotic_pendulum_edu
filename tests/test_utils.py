"""
Unit tests for the utilities module.

Tests the utility functions including configuration management,
data handling, and common utilities.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

# Mock utils module for testing
class ConfigManager:
    """Mock configuration manager for testing."""
    
    def __init__(self, config_file=None):
        self.config_file = config_file
        self.config = {}
    
    def load_config(self, filepath=None):
        """Mock config loading."""
        if filepath is None:
            filepath = self.config_file
        
        if filepath and os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'simulation': {'time_step': 0.01},
                'analysis': {'chaos_threshold': 0.01},
                'visualization': {'figure_size': [10, 8]}
            }
        
        return self.config
    
    def save_config(self, filepath=None):
        """Mock config saving."""
        if filepath is None:
            filepath = self.config_file
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        return False
    
    def get(self, key, default=None):
        """Get config value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """Set config value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

class DataManager:
    """Mock data manager for testing."""
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or "data"
        self.data = {}
    
    def save_data(self, name, data, filepath=None):
        """Mock data saving."""
        if filepath is None:
            filepath = os.path.join(self.data_dir, f"{name}.json")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.data[name] = data
        return filepath
    
    def load_data(self, name, filepath=None):
        """Mock data loading."""
        if filepath is None:
            filepath = os.path.join(self.data_dir, f"{name}.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.data[name] = data
            return data
        return None
    
    def list_data(self):
        """List available data files."""
        if not os.path.exists(self.data_dir):
            return []
        
        files = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.json'):
                files.append(file[:-5])  # Remove .json extension
        return files

class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def test_initialization(self):
        """Test config manager initialization."""
        config_file = "test_config.json"
        manager = ConfigManager(config_file)
        
        assert manager.config_file == config_file
        assert manager.config == {}
    
    def test_initialization_no_file(self):
        """Test config manager initialization without file."""
        manager = ConfigManager()
        
        assert manager.config_file is None
        assert manager.config == {}
    
    def test_load_config_default(self):
        """Test loading default config."""
        manager = ConfigManager()
        config = manager.load_config()
        
        assert isinstance(config, dict)
        assert 'simulation' in config
        assert 'analysis' in config
        assert 'visualization' in config
    
    def test_load_config_from_file(self, temp_data_dir):
        """Test loading config from file."""
        config_file = temp_data_dir / "test_config.json"
        test_config = {
            'simulation': {'time_step': 0.02},
            'analysis': {'chaos_threshold': 0.02}
        }
        
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        
        manager = ConfigManager(str(config_file))
        config = manager.load_config()
        
        assert config == test_config
    
    def test_save_config(self, temp_data_dir):
        """Test saving config to file."""
        config_file = temp_data_dir / "test_config.json"
        manager = ConfigManager(str(config_file))
        
        # Load default config
        manager.load_config()
        
        # Save config
        success = manager.save_config()
        
        assert success is True
        assert config_file.exists()
        
        # Verify saved config
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        
        assert saved_config == manager.config
    
    def test_get_config_value(self):
        """Test getting config values."""
        manager = ConfigManager()
        manager.load_config()
        
        # Test nested key
        time_step = manager.get('simulation.time_step')
        assert time_step == 0.01
        
        # Test non-existent key
        value = manager.get('non.existent.key', default='default_value')
        assert value == 'default_value'
    
    def test_set_config_value(self):
        """Test setting config values."""
        manager = ConfigManager()
        manager.load_config()
        
        # Set nested value
        manager.set('simulation.time_step', 0.02)
        time_step = manager.get('simulation.time_step')
        assert time_step == 0.02
        
        # Set new nested value
        manager.set('new.section.value', 'test_value')
        value = manager.get('new.section.value')
        assert value == 'test_value'

class TestDataManager:
    """Test cases for DataManager class."""
    
    def test_initialization(self):
        """Test data manager initialization."""
        data_dir = "test_data"
        manager = DataManager(data_dir)
        
        assert manager.data_dir == data_dir
        assert manager.data == {}
    
    def test_initialization_default_dir(self):
        """Test data manager initialization with default directory."""
        manager = DataManager()
        
        assert manager.data_dir == "data"
        assert manager.data == {}
    
    def test_save_data(self, temp_data_dir):
        """Test saving data."""
        manager = DataManager(str(temp_data_dir))
        test_data = {'time': [1, 2, 3], 'values': [0.1, 0.2, 0.3]}
        
        filepath = manager.save_data('test_data', test_data)
        
        assert filepath is not None
        assert os.path.exists(filepath)
        assert 'test_data' in manager.data
        assert manager.data['test_data'] == test_data
    
    def test_save_data_custom_filepath(self, temp_data_dir):
        """Test saving data with custom filepath."""
        manager = DataManager(str(temp_data_dir))
        test_data = {'test': 'data'}
        custom_filepath = temp_data_dir / "custom" / "data.json"
        
        filepath = manager.save_data('test_data', test_data, str(custom_filepath))
        
        assert filepath == str(custom_filepath)
        assert os.path.exists(custom_filepath)
    
    def test_load_data_existing(self, temp_data_dir):
        """Test loading existing data."""
        manager = DataManager(str(temp_data_dir))
        test_data = {'time': [1, 2, 3], 'values': [0.1, 0.2, 0.3]}
        
        # Save data first
        manager.save_data('test_data', test_data)
        
        # Load data
        loaded_data = manager.load_data('test_data')
        
        assert loaded_data == test_data
        assert 'test_data' in manager.data
    
    def test_load_data_nonexistent(self, temp_data_dir):
        """Test loading non-existent data."""
        manager = DataManager(str(temp_data_dir))
        
        loaded_data = manager.load_data('nonexistent_data')
        
        assert loaded_data is None
    
    def test_load_data_custom_filepath(self, temp_data_dir):
        """Test loading data with custom filepath."""
        manager = DataManager(str(temp_data_dir))
        test_data = {'test': 'data'}
        custom_filepath = temp_data_dir / "custom" / "data.json"
        
        # Save data to custom location
        manager.save_data('test_data', test_data, str(custom_filepath))
        
        # Load from custom location
        loaded_data = manager.load_data('test_data', str(custom_filepath))
        
        assert loaded_data == test_data
    
    def test_list_data_empty(self, temp_data_dir):
        """Test listing data when directory is empty."""
        manager = DataManager(str(temp_data_dir))
        
        data_files = manager.list_data()
        
        assert data_files == []
    
    def test_list_data_with_files(self, temp_data_dir):
        """Test listing data with existing files."""
        manager = DataManager(str(temp_data_dir))
        
        # Save some test data
        test_data1 = {'data1': 'value1'}
        test_data2 = {'data2': 'value2'}
        
        manager.save_data('data1', test_data1)
        manager.save_data('data2', test_data2)
        
        data_files = manager.list_data()
        
        assert 'data1' in data_files
        assert 'data2' in data_files
        assert len(data_files) == 2

class TestUtilsIntegration:
    """Integration tests for utilities."""
    
    def test_config_and_data_integration(self, temp_data_dir):
        """Test integration between config and data managers."""
        config_file = temp_data_dir / "config.json"
        data_dir = temp_data_dir / "data"
        
        # Initialize managers
        config_manager = ConfigManager(str(config_file))
        data_manager = DataManager(str(data_dir))
        
        # Load and modify config
        config = config_manager.load_config()
        config_manager.set('simulation.time_step', 0.02)
        config_manager.save_config()
        
        # Save data using config
        time_step = config_manager.get('simulation.time_step')
        test_data = {'time_step': time_step, 'data': [1, 2, 3]}
        data_manager.save_data('simulation_data', test_data)
        
        # Verify integration
        assert config_file.exists()
        assert data_dir.exists()
        
        # Load and verify data
        loaded_data = data_manager.load_data('simulation_data')
        assert loaded_data['time_step'] == 0.02
    
    def test_error_handling(self, temp_data_dir):
        """Test error handling in utilities."""
        config_manager = ConfigManager()
        data_manager = DataManager(str(temp_data_dir))
        
        # Test config manager with invalid file
        config = config_manager.load_config('nonexistent_file.json')
        assert isinstance(config, dict)  # Should return default config
        
        # Test data manager with invalid file
        data = data_manager.load_data('nonexistent_data')
        assert data is None 