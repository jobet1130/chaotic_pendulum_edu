import json
import os
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add the parent directory to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.chaos_labeler import ChaosLabeler, main
from src.simulate import simulate


class TestIntegrationChaosLabeler(unittest.TestCase):
    """
    Integration tests for the ChaosLabeler module.

    These tests verify that the ChaosLabeler works correctly with other components
    of the system, particularly focusing on the end-to-end workflow from simulation
    data generation to chaos detection and visualization.

    The tests use a temporary directory structure to isolate test data and outputs,
    ensuring that tests are repeatable and don't interfere with existing data.
    """

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for test files
        cls.temp_dir = Path(os.path.abspath("temp_integration_test"))
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
        cls.temp_dir.mkdir(parents=True)

        # Create subdirectories for data
        cls.raw_dir = cls.temp_dir / "raw"
        cls.processed_dir = cls.temp_dir / "processed"
        cls.labeled_dir = cls.temp_dir / "labeled"
        cls.plots_dir = cls.temp_dir / "plots"

        cls.raw_dir.mkdir()
        cls.processed_dir.mkdir()
        cls.labeled_dir.mkdir()
        cls.plots_dir.mkdir()

        # Create a test configuration file with parameters suitable for testing
        cls.test_config = {
            "gravity": 9.81,
            "length": 1.0,
            "mass": 1.0,
            "damping": 0.05,
            "initial_angle": 0.2,
            "initial_velocity": 0.0,
            "time_step": 0.05,  # Larger time step for faster simulation
            "total_time": 20.0,  # Shorter time for testing
            "drive_amplitude": 1.2,
            "drive_frequency": 2.0,
            "integration_method": "RK4",
            "chaos_detection": {
                "lyapunov_method": "rosenstein",
                "embedding_dimension": 3,
                "time_delay": 2,
                "threshold": 0.1,
            },
            "random_seed": 42,
            "batch_simulations": 3,  # Fewer simulations for testing
            "export_format": "csv",
        }

        cls.config_path = cls.temp_dir / "test_config.json"
        with open(cls.config_path, "w") as f:
            json.dump(cls.test_config, f)

        # Generate simulation data
        cls.simulation_data_path = (
            cls.processed_dir / "chaotic_pendulum_simulations.csv"
        )
        cls.labeled_data_path = cls.labeled_dir / "chaotic_pendulum_labeled.csv"

        # Create simulation data with different drive forces to ensure some chaotic and non-chaotic cases
        cls._generate_test_data()

    @classmethod
    def _generate_test_data(cls):
        """
        Generate test data with a mix of chaotic and non-chaotic pendulum simulations.
        """
        # Create a DataFrame with pendulum simulations at different drive forces
        time_points = np.arange(
            0, cls.test_config["total_time"], cls.test_config["time_step"]
        )
        num_points = len(time_points)

        # Use different drive forces to ensure a mix of chaotic and non-chaotic behavior
        drive_forces = [
            -1.5,
            -0.5,
            0.8,
            1.2,
            1.5,
        ]  # Mix of values likely to produce different behaviors

        data_list = []
        for c in drive_forces:
            # Generate synthetic pendulum data
            # For chaotic pendulum, higher drive forces typically lead to chaos
            # We'll use a simple model here for testing purposes

            # Seed for reproducibility
            np.random.seed(42 + int(c * 10))

            # Generate angle data with increasing complexity for higher drive forces
            if abs(c) > 1.0:  # More likely to be chaotic
                # More complex, potentially chaotic motion
                theta = 0.1 * np.sin(time_points) + 0.1 * np.sin(2.5 * time_points)
                theta += 0.05 * np.sin(3.7 * time_points) + 0.02 * np.random.randn(
                    num_points
                )
                omega = np.gradient(theta, cls.test_config["time_step"])
            else:  # More likely to be regular
                # Simple periodic motion
                theta = 0.2 * np.sin(time_points) + 0.05 * np.sin(3 * time_points)
                omega = np.gradient(theta, cls.test_config["time_step"])

            # Create DataFrame for this drive force
            df = pd.DataFrame(
                {
                    "num__t": time_points,
                    "num__theta": theta,
                    "num__omega": omega,
                    "num__drive_force_c": np.full(num_points, c),
                }
            )

            data_list.append(df)

        # Combine all simulations
        combined_df = pd.concat(data_list, ignore_index=True)

        # Save to CSV
        combined_df.to_csv(cls.simulation_data_path, index=False)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directory
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    def test_end_to_end_workflow(self):
        """
        Test the complete workflow from data loading to chaos detection and visualization.
        """
        # Initialize ChaosLabeler with test config
        labeler = ChaosLabeler(config_path=str(self.config_path))

        # Load simulation data
        df = labeler.load_data(self.simulation_data_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        # Detect chaos using both Lyapunov exponent and correlation dimension
        labeled_df = labeler.detect_chaos(df, use_correlation_dim=True)
        self.assertIn("is_chaotic", labeled_df.columns)

        # Verify that chaos detection produced results
        unique_forces = labeled_df[
            ["num__drive_force_c", "is_chaotic"]
        ].drop_duplicates()
        self.assertGreater(
            unique_forces["is_chaotic"].sum(), 0, "No chaotic cases detected"
        )

        # Note: We don't assert that some cases must be non-chaotic, as the detection algorithm
        # might legitimately classify all our test cases as chaotic depending on the parameters

        # Save labeled data
        labeler.save_labeled_data(self.labeled_data_path)
        self.assertTrue(self.labeled_data_path.exists())

        # Generate phase space plots
        labeler.plot_phase_space(labeled_df, save_dir=self.plots_dir, plot_3d=True)

        # Check that plots were generated
        plot_files = list(self.plots_dir.glob("*.png"))
        self.assertGreater(len(plot_files), 0, "No plot files were generated")

        # Verify both 2D and 3D plots were created
        plot_2d_files = list(self.plots_dir.glob("phase_space_c_*.png"))
        plot_3d_files = list(self.plots_dir.glob("phase_space_3d_c_*.png"))
        self.assertGreater(
            len(plot_2d_files), 0, "No 2D phase space plots were generated"
        )
        self.assertGreater(
            len(plot_3d_files), 0, "No 3D phase space plots were generated"
        )

    def test_main_function_integration(self):
        """
        Test the main function with command-line arguments.
        """
        # Prepare arguments for main function
        import sys
        from unittest.mock import patch

        test_args = [
            "chaos_labeler.py",
            "--data",
            str(self.simulation_data_path),
            "--config",
            str(self.config_path),
            "--output",
            str(self.labeled_dir / "main_output.csv"),
            "--plot-dir",
            str(self.plots_dir / "main_plots"),
            "--use-corr-dim",
            "--column",
            "num__theta",
        ]

        # Create the output directory
        (self.plots_dir / "main_plots").mkdir(exist_ok=True)

        # Mock sys.argv and run main
        with patch("sys.argv", test_args):
            # Import the module here to ensure we're using the patched sys.argv
            import src.chaos_labeler

            src.chaos_labeler.main()

        # Check that output file was created
        output_path = self.labeled_dir / "main_output.csv"
        self.assertTrue(output_path.exists())

        # Check that plots were generated
        plot_files = list((self.plots_dir / "main_plots").glob("*.png"))
        self.assertGreater(
            len(plot_files), 0, "No plot files were generated by main function"
        )

    def test_integration_with_different_columns(self):
        """
        Test chaos detection using different columns (theta vs omega).
        """
        labeler = ChaosLabeler(config_path=str(self.config_path))
        df = labeler.load_data(self.simulation_data_path)

        # Detect chaos using theta column
        labeled_theta = labeler.detect_chaos(
            df, column="num__theta", use_correlation_dim=True
        )

        # Detect chaos using omega column
        labeled_omega = labeler.detect_chaos(
            df, column="num__omega", use_correlation_dim=True
        )

        # Compare results - they might differ as chaos can manifest differently in position vs velocity
        theta_chaos = labeled_theta[
            ["num__drive_force_c", "is_chaotic"]
        ].drop_duplicates()
        omega_chaos = labeled_omega[
            ["num__drive_force_c", "is_chaotic"]
        ].drop_duplicates()

        # We're not asserting they should be the same, just that both analyses completed successfully
        self.assertEqual(
            len(theta_chaos),
            len(omega_chaos),
            "Analysis of theta and omega should cover the same drive forces",
        )


if __name__ == "__main__":
    unittest.main()
