import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add the parent directory to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.chaos_classifier import ChaosClassifier, main


class TestChaosClassifier(unittest.TestCase):
    """
    Test case for the ChaosClassifier class.

    This test case covers the functionality of the ChaosClassifier class, including:
    - Configuration loading
    - Data loading and processing
    - Model training and evaluation
    - Model saving and loading
    - Prediction functionality
    """

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()

        # Create a test configuration file
        cls.test_config = {
            "gravity": 9.81,
            "length": 1.0,
            "mass": 1.0,
            "damping": 0.05,
            "initial_angle": 0.2,
            "initial_velocity": 0.0,
            "time_step": 0.01,
            "total_time": 10.0,
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
            "batch_simulations": 2,
            "export_format": "csv",
        }

        cls.config_path = Path(cls.temp_dir) / "test_config.json"
        with open(cls.config_path, "w") as f:
            json.dump(cls.test_config, f)

        # Create test labeled data
        cls.test_data = pd.DataFrame(
            {
                "num__t": np.linspace(0, 10, 100),
                "num__theta": np.sin(np.linspace(0, 10, 100)),
                "num__omega": np.cos(np.linspace(0, 10, 100)),
                "num__drive_force_c": np.concatenate(
                    [np.full(50, 0.5), np.full(50, 1.2)]
                ),
                "is_chaotic": np.concatenate([np.full(50, False), np.full(50, True)]),
            }
        )

        cls.data_path = Path(cls.temp_dir) / "test_labeled_data.csv"
        cls.test_data.to_csv(cls.data_path, index=False)

        # Create output directories
        cls.models_dir = Path(cls.temp_dir) / "models"
        cls.models_dir.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directory
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        # Create a fresh instance for each test
        self.classifier = ChaosClassifier(config_path=str(self.config_path))

    def test_init_and_load_config(self):
        """Test initialization and configuration loading."""
        # Test that config is loaded correctly
        self.assertEqual(self.classifier.config["gravity"], 9.81)
        self.assertEqual(self.classifier.config["damping"], 0.05)
        self.assertEqual(
            self.classifier.config["chaos_detection"]["embedding_dimension"], 3
        )

        # Test with non-existent config file
        with self.assertRaises(FileNotFoundError):
            ChaosClassifier(config_path="non_existent_config.json")

    def test_load_data(self):
        """Test data loading functionality."""
        # Test loading valid data
        df = self.classifier.load_data(self.data_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)

        # Test with non-existent data file
        with self.assertRaises(FileNotFoundError):
            self.classifier.load_data("non_existent_data.csv")

    def test_prepare_features(self):
        """Test feature preparation functionality."""
        df = self.classifier.load_data(self.data_path)
        X, y = self.classifier.prepare_features(df)

        # Check that features and target are correctly extracted
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), 100)
        self.assertEqual(len(y), 100)

        # Check that feature builder is created
        self.assertIsNotNone(self.classifier.feature_builder)

        # Test with missing target column
        df_no_target = df.drop(columns=["is_chaotic"])
        with self.assertRaises(ValueError):
            self.classifier.prepare_features(df_no_target)

    def test_train_model(self):
        """Test model training functionality."""
        # Prepare data
        df = self.classifier.load_data(self.data_path)
        X, y = self.classifier.prepare_features(df)

        # Create a simplified version of train_model that doesn't use GridSearchCV
        with patch("src.chaos_classifier.GridSearchCV") as mock_grid_search:
            # Mock the grid search to avoid actual training
            mock_grid_search_instance = MagicMock()
            mock_grid_search.return_value = mock_grid_search_instance
            mock_grid_search_instance.best_estimator_ = RandomForestClassifier()
            mock_grid_search_instance.best_params_ = {"classifier__n_estimators": 100}

            # Mock fit to avoid actual training
            mock_grid_search_instance.fit = MagicMock()

            # Mock predict to return valid predictions
            mock_grid_search_instance.best_estimator_.predict = MagicMock(
                return_value=np.zeros(20)
            )

            # Patch train_test_split to return small arrays
            with patch("src.chaos_classifier.train_test_split") as mock_split:
                # Return small subsets to avoid computation
                # Make sure the test set size matches the prediction array size
                mock_split.return_value = (
                    X.iloc[:20],
                    X.iloc[20:40],
                    y.iloc[:20],
                    y.iloc[20:40],
                )

                # Train model
                self.classifier.train_model(X, y, cv=2, n_jobs=1)

                # Check that grid search was called
                mock_grid_search.assert_called_once()
                mock_grid_search_instance.fit.assert_called_once()

                # Check that model is set
                self.assertIsNotNone(self.classifier.model)

    def test_save_and_load_model(self):
        """Test model saving and loading functionality."""
        # Create a mock model
        self.classifier.model = MagicMock()

        # Test saving model
        model_path = self.models_dir / "test_model.joblib"
        with patch("joblib.dump") as mock_dump:
            self.classifier.save_model(model_path)
            mock_dump.assert_called_once()

        # Test loading model
        with patch("joblib.load", return_value=MagicMock()) as mock_load:
            with patch("pathlib.Path.exists", return_value=True):
                self.classifier.load_model(model_path)
                mock_load.assert_called_once()

        # Test saving with no model
        self.classifier.model = None
        with self.assertRaises(RuntimeError):
            self.classifier.save_model(model_path)

        # Test loading non-existent model
        with self.assertRaises(FileNotFoundError):
            self.classifier.load_model("non_existent_model.joblib")

    def test_predict(self):
        """Test prediction functionality."""
        # Create a mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([True, False])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        self.classifier.model = mock_model

        # Test predict
        X = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        predictions = self.classifier.predict(X)
        mock_model.predict.assert_called_once()
        self.assertEqual(len(predictions), 2)

        # Test predict_proba
        probabilities = self.classifier.predict_proba(X)
        mock_model.predict_proba.assert_called_once()
        self.assertEqual(probabilities.shape, (2, 2))

        # Test predict with no model
        self.classifier.model = None
        with self.assertRaises(RuntimeError):
            self.classifier.predict(X)

        # Test predict_proba with no model
        with self.assertRaises(RuntimeError):
            self.classifier.predict_proba(X)

    def test_main_function(self):
        """Test the main function."""
        # Test the training workflow
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            # Mock command line arguments for training
            mock_args = MagicMock()
            mock_args.data = str(self.data_path)
            mock_args.config = str(self.config_path)
            mock_args.model = os.path.join(self.models_dir, "main_model.joblib")
            mock_args.train = True
            mock_args.predict = False
            mock_args.output = None
            mock_parse_args.return_value = mock_args

            # Create a mock DataFrame with is_chaotic column
            mock_df = pd.DataFrame(
                {
                    "feature1": np.arange(100),
                    "feature2": np.arange(100),
                    "is_chaotic": np.array([True] * 50 + [False] * 50),
                }
            )

            # Mock the classifier methods
            with patch(
                "src.chaos_classifier.ChaosClassifier.load_data", return_value=mock_df
            ):
                with patch(
                    "src.chaos_classifier.ChaosClassifier.prepare_features"
                ) as mock_prepare:
                    # Return actual DataFrame and Series to avoid type errors
                    mock_prepare.return_value = (
                        mock_df.drop(columns=["is_chaotic"]),
                        mock_df["is_chaotic"],
                    )

                    with patch("src.chaos_classifier.ChaosClassifier.train_model"):
                        with patch("src.chaos_classifier.ChaosClassifier.save_model"):
                            # Run main function
                            main()

        # Test the prediction workflow
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            # Mock command line arguments for prediction
            mock_args = MagicMock()
            mock_args.data = str(self.data_path)
            mock_args.config = str(self.config_path)
            mock_args.model = os.path.join(self.models_dir, "main_model.joblib")
            mock_args.train = False
            mock_args.predict = True
            mock_args.output = os.path.join(self.temp_dir, "predictions.csv")
            mock_parse_args.return_value = mock_args

            # Create a mock DataFrame without is_chaotic column for prediction
            mock_df = pd.DataFrame(
                {"feature1": np.arange(100), "feature2": np.arange(100)}
            )

            # Mock the classifier methods for prediction
            with patch("src.chaos_classifier.ChaosClassifier.load_model"):
                with patch(
                    "src.chaos_classifier.ChaosClassifier.load_data",
                    return_value=mock_df,
                ):
                    with patch(
                        "src.chaos_classifier.ChaosClassifier.prepare_features"
                    ) as mock_prepare:
                        # Return actual DataFrame and None to avoid type errors
                        mock_prepare.return_value = (mock_df, None)

                        with patch(
                            "src.chaos_classifier.ChaosClassifier.predict"
                        ) as mock_predict:
                            # Return numpy array of predictions
                            mock_predict.return_value = np.array(
                                [True] * 50 + [False] * 50
                            )

                            with patch(
                                "src.chaos_classifier.ChaosClassifier.predict_proba"
                            ) as mock_proba:
                                # Return numpy array of probabilities
                                mock_proba.return_value = np.column_stack(
                                    (np.random.random(100), np.random.random(100))
                                )

                                # Mock DataFrame.to_csv to avoid file operations
                                with patch("pandas.DataFrame.to_csv"):
                                    # Run main function
                                    main()


if __name__ == "__main__":
    unittest.main()
