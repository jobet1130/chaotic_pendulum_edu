import logging
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


class FeatureBuilder:
    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        drop_features: Optional[List[str]] = None,
    ):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.drop_features = drop_features or []
        self.pipeline = None

    def _build_pipeline(self) -> ColumnTransformer:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.numerical_features),
                ("cat", cat_pipeline, self.categorical_features),
            ],
            remainder="drop",
        )
        return transformer

    def fit(self, df: pd.DataFrame) -> None:
        logging.info("Fitting feature transformer pipeline.")
        print("🔧 Fitting feature transformer pipeline...")
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(df)
        logging.info("Pipeline fit complete.")
        print("✅ Pipeline fit complete.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Pipeline has not been fitted yet.")

        logging.info("Transforming dataset.")
        print("🔄 Transforming dataset...")
        transformed_data = self.pipeline.transform(df)
        transformed_df = pd.DataFrame(
            transformed_data.toarray()
            if hasattr(transformed_data, "toarray")
            else transformed_data
        )
        logging.info("Transformation complete.")
        print("✅ Transformation complete.")
        return transformed_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save_pipeline(self, path: Path) -> None:
        if self.pipeline is None:
            raise RuntimeError("No pipeline to save.")
        joblib.dump(self.pipeline, path)
        logging.info(f"Pipeline saved to {path}")
        print(f"💾 Pipeline saved to {path}")

    def load_pipeline(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")
        self.pipeline = joblib.load(path)
        logging.info(f"Pipeline loaded from {path}")
        print(f"📂 Pipeline loaded from {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature engineering pipeline runner.")
    parser.add_argument(
        "--raw_dir", type=str, required=True, help="Path to directory containing raw CSV files."
    )
    parser.add_argument(
        "--processed_dir", type=str, required=True, help="Path to directory to save processed CSV files."
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="pipeline.joblib",
        help="Path to save/load pipeline joblib file.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in raw_dir.glob("*.csv"):
        print(f"\n📂 Processing: {csv_file.name}")
        df = pd.read_csv(csv_file)

        numerical = df.select_dtypes(include="number").columns.tolist()
        categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

        fb = FeatureBuilder(numerical, categorical)
        transformed_df = fb.fit_transform(df)

        output_file = processed_dir / csv_file.name
        transformed_df.to_csv(output_file, index=False)
        print(f"✅ Processed CSV saved to: {output_file}")

        fb.save_pipeline(Path(args.pipeline))