import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.utils import get_csv_files, mark_csvs_as_processed


class TestCSVProcessor(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for raw and processed data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.temp_dir.name) / "raw"
        self.processed_dir = Path(self.temp_dir.name) / "processed"

        self.raw_dir.mkdir()
        self.processed_dir.mkdir()

        # Create sample CSV files in raw_dir
        self.sample_files = [
            self.raw_dir / "data1.csv",
            self.raw_dir / "data2.csv",
        ]
        for file in self.sample_files:
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            df.to_csv(file, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_csv_files(self):
        csv_files = get_csv_files(self.raw_dir)
        self.assertEqual(len(csv_files), 2)
        self.assertTrue(all(file.suffix == ".csv" for file in csv_files))

    def test_mark_csvs_as_processed_copy(self):
        mark_csvs_as_processed(
            raw_dir=str(self.raw_dir),
            processed_dir=str(self.processed_dir),
            delete_original=False,
        )

        processed_files = list(self.processed_dir.glob("*.csv"))
        self.assertEqual(len(processed_files), 2)

        for file in self.sample_files:
            self.assertTrue(file.exists())  # Originals should remain

    def test_mark_csvs_as_processed_delete(self):
        mark_csvs_as_processed(
            raw_dir=str(self.raw_dir),
            processed_dir=str(self.processed_dir),
            delete_original=True,
        )

        processed_files = list(self.processed_dir.glob("*.csv"))
        self.assertEqual(len(processed_files), 2)

        for file in self.sample_files:
            self.assertFalse(file.exists())  # Originals should be deleted

    def test_no_csv_files(self):
        # Clear raw_dir
        for file in self.sample_files:
            file.unlink()

        csv_files = get_csv_files(self.raw_dir)
        self.assertEqual(csv_files, [])

        # Ensure it doesn't crash
        mark_csvs_as_processed(
            raw_dir=str(self.raw_dir),
            processed_dir=str(self.processed_dir),
        )


if __name__ == "__main__":
    unittest.main()
