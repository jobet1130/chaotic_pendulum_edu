import shutil
import unittest
from pathlib import Path

import pandas as pd

from src.features import FeatureBuilder


class TestFeatureBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = pd.DataFrame(
            {
                "theta": [0.1, 0.2, 0.3],
                "omega": [0.01, 0.02, 0.03],
                "drive_force_c": [0.5, 0.6, 0.7],
                "category": ["a", "b", "a"],
            }
        )
        cls.numerical = ["theta", "omega", "drive_force_c"]
        cls.categorical = ["category"]
        cls.pipeline_path = Path("models/test_pipeline.joblib")

        # Ensure clean model folder
        if cls.pipeline_path.parent.exists():
            shutil.rmtree(cls.pipeline_path.parent)
        cls.pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    def test_fit_and_transform(self):
        fb = FeatureBuilder(self.numerical, self.categorical)
        transformed_df = fb.fit_transform(self.test_data)

        # Expect transformed output to be a DataFrame
        self.assertIsInstance(transformed_df, pd.DataFrame)
        self.assertGreaterEqual(transformed_df.shape[1], len(self.numerical))

    def test_save_and_load_pipeline(self):
        fb = FeatureBuilder(self.numerical, self.categorical)
        fb.fit(self.test_data)
        fb.save_pipeline(self.pipeline_path)

        self.assertTrue(self.pipeline_path.exists(), "Pipeline file was not saved.")

        # Load into a new instance and transform
        new_fb = FeatureBuilder(self.numerical, self.categorical)
        new_fb.load_pipeline(self.pipeline_path)
        transformed_df = new_fb.transform(self.test_data)

        self.assertIsInstance(transformed_df, pd.DataFrame)

    @classmethod
    def tearDownClass(cls):
        if cls.pipeline_path.parent.exists():
            shutil.rmtree(cls.pipeline_path.parent)


if __name__ == "__main__":
    unittest.main()
