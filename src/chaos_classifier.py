import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.features import FeatureBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


class ChaosClassifier:
    """Class for classifying chaotic pendulum data.

    This class provides methods to train machine learning models that can predict
    whether a pendulum system is chaotic or non-chaotic based on its parameters
    and time series data. It uses the labeled data from ChaosLabeler as training data.
    """

    def __init__(self, config_path: str = "pendulum_config.json"):
        """Initialize the ChaosClassifier with configuration.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_builder = None

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            Dict: Configuration parameters.
        """
        with open(config_path, "r") as file:
            return json.load(file)

    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Load labeled pendulum data.

        Args:
            data_path (Union[str, Path]): Path to the labeled data file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logging.info(f"Loading labeled data from {data_path}")
        print(f"📊 Loading labeled data from {data_path}")

        df = pd.read_csv(data_path)
        return df

    def prepare_features(
        self, df: pd.DataFrame, target_column: str = "is_chaotic"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model training.

        Args:
            df (pd.DataFrame): Labeled data.
            target_column (str): Name of the target column.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")

        # Identify numerical and categorical features
        # Exclude the target column and any metadata columns
        exclude_columns = [
            target_column,
            "drive_force_c",
            "lyapunov_exponent",
            "correlation_dimension",
            "is_chaotic_lyapunov",
            "is_chaotic_corr_dim",
        ]
        feature_columns = [col for col in df.columns if col not in exclude_columns]

        numerical_features = (
            df[feature_columns].select_dtypes(include=["number"]).columns.tolist()
        )
        categorical_features = (
            df[feature_columns]
            .select_dtypes(include=["object", "category"])
            .columns.tolist()
        )

        logging.info(f"Numerical features: {numerical_features}")
        logging.info(f"Categorical features: {categorical_features}")

        # Create feature builder
        self.feature_builder = FeatureBuilder(numerical_features, categorical_features)

        # Extract features and target
        X = df[feature_columns]
        y = df[target_column]

        return X, y

    def train_model(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 5, n_jobs: int = -1
    ) -> None:
        """Train a machine learning model to classify chaotic behavior.

        Args:
            X (pd.DataFrame): Feature data.
            y (pd.Series): Target labels.
            cv (int): Number of cross-validation folds.
            n_jobs (int): Number of parallel jobs for grid search.
        """
        logging.info("Training chaos classification model...")
        print("🧠 Training chaos classification model...")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config.get("random_seed", 42)
        )

        # Create a pipeline with feature transformation and classifier
        pipeline = Pipeline(
            [
                ("features", self.feature_builder.pipeline),
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=self.config.get("random_seed", 42)
                    ),
                ),
            ]
        )

        # Define hyperparameter grid
        param_grid = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
        }

        # Perform grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, n_jobs=n_jobs, scoring="accuracy", verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Get best model
        self.model = grid_search.best_estimator_

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logging.info(f"Best parameters: {grid_search.best_params_}")
        logging.info(f"Model accuracy: {accuracy:.4f}")

        print(f"✅ Model trained with accuracy: {accuracy:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def save_model(self, model_path: Union[str, Path]) -> None:
        """Save the trained model to disk.

        Args:
            model_path (Union[str, Path]): Path to save the model.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train a model first.")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_path)
        logging.info(f"Model saved to {model_path}")
        print(f"💾 Model saved to {model_path}")

    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load a trained model from disk.

        Args:
            model_path (Union[str, Path]): Path to the saved model.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        print(f"📂 Model loaded from {model_path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict chaotic behavior for new data.

        Args:
            X (pd.DataFrame): Feature data.

        Returns:
            np.ndarray: Predicted labels (True for chaotic, False for non-chaotic).
        """
        if self.model is None:
            raise RuntimeError("No model available. Train or load a model first.")

        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of chaotic behavior for new data.

        Args:
            X (pd.DataFrame): Feature data.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            raise RuntimeError("No model available. Train or load a model first.")

        probabilities = self.model.predict_proba(X)
        return probabilities


def main():
    """Main function for command-line execution.

    This function handles command-line arguments and executes the appropriate
    actions based on the provided arguments. It supports training a new model
    or using an existing model to make predictions.
    """
    parser = argparse.ArgumentParser(description="Chaos Classifier for Pendulum Data")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to labeled data file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pendulum_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model", type=str, help="Path to model file (for saving or loading)"
    )
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Make predictions using an existing model",
    )
    parser.add_argument("--output", type=str, help="Path to save prediction results")

    args = parser.parse_args()

    # Initialize classifier
    classifier = ChaosClassifier(config_path=args.config)

    # Load data
    df = classifier.load_data(args.data)

    if args.train:
        # Train model
        X, y = classifier.prepare_features(df)
        classifier.train_model(X, y)

        # Save model if path is provided
        if args.model:
            classifier.save_model(args.model)

    elif args.predict:
        # Load model
        if not args.model:
            raise ValueError("Model path must be provided for prediction.")

        classifier.load_model(args.model)

        # Prepare features
        X, _ = classifier.prepare_features(
            df, target_column="is_chaotic" if "is_chaotic" in df.columns else None
        )

        # Make predictions
        predictions = classifier.predict(X)
        probabilities = classifier.predict_proba(X)

        # Add predictions to dataframe
        df["predicted_chaotic"] = predictions
        df["chaotic_probability"] = probabilities[:, 1]  # Probability of being chaotic

        # Save results if output path is provided
        if args.output:
            df.to_csv(args.output, index=False)
            logging.info(f"Predictions saved to {args.output}")
            print(f"💾 Predictions saved to {args.output}")
        else:
            # Print summary of predictions
            chaotic_count = predictions.sum()
            total_count = len(predictions)
            print(f"\nPrediction Summary:")
            print(f"Total samples: {total_count}")
            print(f"Chaotic samples: {chaotic_count} ({chaotic_count/total_count:.2%})")
            print(
                f"Non-chaotic samples: {total_count - chaotic_count} ({(total_count - chaotic_count)/total_count:.2%})"
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
