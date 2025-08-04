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

    def test_get_csv_files_nonexistent_directory(self):
        """Test getting CSV files from a non-existent directory."""
        # Create a path to a non-existent directory
        nonexistent_dir = Path(self.temp_dir.name) / "nonexistent"

        # Get CSV files from non-existent directory
        with self.assertLogs(level="WARNING") as cm:
            csv_files = get_csv_files(nonexistent_dir)

            # Check that a warning was logged
            self.assertTrue(any("Directory not found" in log for log in cm.output))

        # Check that an empty list is returned
        self.assertEqual(len(csv_files), 0)

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

    def test_mark_csvs_as_processed_with_overwrite(self):
        """Test marking CSV files as processed with overwrite."""
        # Create a file in the processed directory with the same name
        processed_file = self.processed_dir / self.sample_files[0].name
        processed_file.touch()

        # Mark CSV files as processed with overwrite warning
        with self.assertLogs(level="WARNING") as cm:
            mark_csvs_as_processed(
                raw_dir=str(self.raw_dir),
                processed_dir=str(self.processed_dir),
                delete_original=False,
            )

            # Check that a warning was logged
            self.assertTrue(any("File will be overwritten" in log for log in cm.output))

        # Check that file was overwritten
        self.assertTrue(processed_file.exists())

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

    def test_mark_csvs_as_processed_error_handling(self):
        """Test error handling in mark_csvs_as_processed."""
        # Create test CSV file
        test_file = self.raw_dir / "error_test.csv"
        test_file.touch()

        # Mock shutil.copy2 to raise an exception
        with (
            unittest.mock.patch("shutil.copy2", side_effect=Exception("Test error")),
            self.assertLogs(level="ERROR") as cm,
        ):

            mark_csvs_as_processed(
                raw_dir=str(self.raw_dir),
                processed_dir=str(self.processed_dir),
                delete_original=True,
            )

            # Check that error was logged
            self.assertTrue(any("Error processing" in log for log in cm.output))

            # Original file should still exist since the copy failed
            self.assertTrue(test_file.exists())


if __name__ == "__main__":
    unittest.main()
