import json
import os
import shutil
import unittest
from pathlib import Path

import pandas as pd

from src.simulate import load_config, simulate


class TestSimulate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_config = {
            "damping": 0.5,
            "total_time": 10,
            "time_step": 0.05,
            "initial_angle": 0.2,
            "initial_velocity": 0.0,
            "batch_simulations": 3,
            "plot_sample": False,
        }
        cls.output_dir = Path("data/raw")
        cls.output_file = cls.output_dir / "chaotic_pendulum_simulations.csv"

        # Ensure clean test environment
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)

    def test_simulation_output(self):
        simulate(self.test_config)

        # Check if file exists
        self.assertTrue(self.output_file.exists(), "Output CSV not found.")

        # Load and verify the content
        df = pd.read_csv(self.output_file)
        self.assertIn("t", df.columns)
        self.assertIn("theta", df.columns)
        self.assertIn("omega", df.columns)
        self.assertIn("drive_force_c", df.columns)

        # Check that there is data
        self.assertGreater(len(df), 0, "CSV file is empty.")

    def test_load_config_file(self):
        config_path = "pendulum_config.json"
        expected_config = {
            "damping": 0.2,
            "total_time": 5,
            "time_step": 0.1,
            "initial_angle": 0.1,
            "initial_velocity": 0.0,
            "batch_simulations": 2,
            "plot_sample": False,
        }

        # Create config file if it doesn't exist
        with open(config_path, "w") as f:
            json.dump(expected_config, f)

        config = load_config(config_path)
        self.assertIn("damping", config)
        self.assertEqual(config["damping"], expected_config["damping"])

    @classmethod
    def tearDownClass(cls):
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)
        config_path = "pendulum_config.json"
        if os.path.exists(config_path):
            os.remove(config_path)


if __name__ == "__main__":
    unittest.main()
