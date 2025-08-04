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
from src.chaos_classifier import ChaosClassifier
from src.chaos_labeler import ChaosLabeler
from src.features import FeatureBuilder


class TestIntegrationChaosClassifier(unittest.TestCase):
    """
    Integration tests for the ChaosClassifier module.

    These tests verify that the ChaosClassifier works correctly with other components
    of the system, particularly focusing on the end-to-end workflow from labeled data
    to model training, evaluation, and prediction.

    The tests use a temporary directory structure to isolate test data and outputs,
    ensuring that tests are repeatable and don't interfere with existing data.
    """

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory structure
        cls.temp_dir = Path(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "temp_integration_test")
            )
        )
        cls.temp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        cls.labeled_dir = cls.temp_dir / "labeled"
        cls.labeled_dir.mkdir(exist_ok=True)

        cls.models_dir = cls.temp_dir / "models"
        cls.models_dir.mkdir(exist_ok=True)

        # Create a test configuration file
        cls.test_config = {
            "gravity": 9.81,
            "length": 1.0,
            "mass": 1.0,
            "damping": 0.05,
            "initial_angle": 0.2,
            "initial_velocity": 0.0,
            "time_step": 0.01,
            "total_time": 10.0,  # Shorter time for testing
            "drive_amplitude": 1.2,
            "drive_frequency": 2.0,
            "integration_method": "RK4",
            "chaos_detection": {
                "lyapunov_method": "rosenstein",
                "embedding_dimension": 3,
                "time_delay": 2,  # Smaller delay for testing
                "threshold": 0.1,
            },
            "random_seed": 42,
            "batch_simulations": 3,  # Fewer simulations for testing
            "export_format": "csv",
        }

        cls.config_path = cls.temp_dir / "test_config.json"
        with open(cls.config_path, "w") as f:
            json.dump(cls.test_config, f)

        # Generate labeled data
        cls.labeled_data_path = cls.labeled_dir / "chaotic_pendulum_labeled.csv"
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
                is_chaotic = True
            else:  # More likely to be regular
                # Simple, regular motion
                theta = 0.1 * np.sin(time_points)
                omega = 0.1 * np.cos(time_points)
                is_chaotic = False

            # Create a DataFrame for this drive force
            df = pd.DataFrame(
                {
                    "num__t": time_points,
                    "num__theta": theta,
                    "num__omega": omega,
                    "num__drive_force_c": np.full(num_points, c),
                    "drive_force_c": np.full(num_points, c),
                    "is_chaotic": np.full(num_points, is_chaotic),
                }
            )

            data_list.append(df)

        # Combine all drive forces into one DataFrame
        combined_df = pd.concat(data_list, ignore_index=True)

        # Save to CSV
        combined_df.to_csv(cls.labeled_data_path, index=False)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directory
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    def test_end_to_end_workflow(self):
        """
        Test the complete workflow from labeled data to model training and prediction.
        """
        # Initialize ChaosClassifier with test config
        classifier = ChaosClassifier(config_path=str(self.config_path))

        # Load labeled data
        df = classifier.load_data(self.labeled_data_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        # Prepare features and target
        X, y = classifier.prepare_features(df)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

        # Train model
        classifier.train_model(X, y, cv=2, n_jobs=1)  # Use small CV for testing
        self.assertIsNotNone(classifier.model)

        # Save model
        model_path = self.models_dir / "chaos_classifier.joblib"
        classifier.save_model(model_path)
        self.assertTrue(model_path.exists())

        # Create a new classifier instance and load the model
        new_classifier = ChaosClassifier(config_path=str(self.config_path))
        new_classifier.load_model(model_path)

        # Make predictions on the same data
        predictions = new_classifier.predict(X)
        probabilities = new_classifier.predict_proba(X)

        # Verify predictions
        self.assertEqual(len(predictions), len(df))
        self.assertEqual(probabilities.shape[0], len(df))
        self.assertEqual(probabilities.shape[1], 2)  # Binary classification

    def test_integration_with_chaos_labeler(self):
        """
        Test integration between ChaosLabeler and ChaosClassifier.
        """
        # Initialize ChaosLabeler and ChaosClassifier
        labeler = ChaosLabeler(config_path=str(self.config_path))
        classifier = ChaosClassifier(config_path=str(self.config_path))

        # Load data with ChaosLabeler
        df = labeler.load_data(self.labeled_data_path)

        # Detect chaos using ChaosLabeler
        labeled_df = labeler.detect_chaos(df, use_correlation_dim=True)
        # The ChaosLabeler might rename columns with suffixes during merging
        self.assertTrue(any(col.startswith("is_chaotic") for col in labeled_df.columns))

        # Save labeled data
        labeled_output_path = self.labeled_dir / "chaotic_pendulum_relabeled.csv"
        labeler.save_labeled_data(labeled_output_path)

        # Use the labeled data for training ChaosClassifier
        df = classifier.load_data(labeled_output_path)

        # Ensure we have a single 'is_chaotic' column for classification
        # Find the column that starts with 'is_chaotic'
        chaotic_cols = [col for col in df.columns if col.startswith("is_chaotic")]
        if len(chaotic_cols) > 0 and "is_chaotic" not in df.columns:
            # Use the first matching column and rename it
            df["is_chaotic"] = df[chaotic_cols[0]]

        X, y = classifier.prepare_features(df)

        # Train model
        classifier.train_model(X, y, cv=2, n_jobs=1)  # Use small CV for testing
        self.assertIsNotNone(classifier.model)

        # Make predictions
        predictions = classifier.predict(X)
        self.assertEqual(len(predictions), len(df))

    def test_integration_with_feature_builder(self):
        """
        Test integration between FeatureBuilder and ChaosClassifier.
        """
        # Load labeled data
        df = pd.read_csv(self.labeled_data_path)

        # Identify numerical and categorical features
        numerical_features = df.select_dtypes(include=["number"]).columns.tolist()
        numerical_features = [
            col
            for col in numerical_features
            if col not in ["is_chaotic", "drive_force_c"]
        ]
        categorical_features = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Create feature builder directly
        feature_builder = FeatureBuilder(numerical_features, categorical_features)

        # Fit and transform data
        transformed_df = feature_builder.fit_transform(df)

        # Save pipeline
        pipeline_path = self.models_dir / "feature_pipeline.joblib"
        feature_builder.save_pipeline(pipeline_path)

        # Initialize ChaosClassifier
        classifier = ChaosClassifier(config_path=str(self.config_path))

        # Prepare features and target using the classifier
        X, y = classifier.prepare_features(df)

        # Verify that the feature builder was created correctly
        self.assertIsNotNone(classifier.feature_builder)

        # Train model
        classifier.train_model(X, y, cv=2, n_jobs=1)  # Use small CV for testing
        self.assertIsNotNone(classifier.model)


if __name__ == "__main__":
    unittest.main()
