import logging
import shutil
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def get_csv_files(directory: Path) -> List[Path]:
    """
    Returns a list of CSV files in the given directory.

    Args:
        directory (Path): Path to search for CSV files.

    Returns:
        List[Path]: List of CSV file paths.
    """
    if not directory.exists() or not directory.is_dir():
        logging.warning(f"Directory not found: {directory}")
        return []
    return list(directory.glob("*.csv"))


def mark_csvs_as_processed(
    raw_dir: str = "../data/raw",
    processed_dir: str = "../data/processed",
    delete_original: bool = False,
) -> None:
    """
    Copies all CSV files from the raw directory to the processed directory.

    Args:
        raw_dir (str): Source directory containing raw CSVs.
        processed_dir (str): Destination directory for processed CSVs.
        delete_original (bool): Whether to delete original files after copying.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    csv_files = get_csv_files(raw_path)
    if not csv_files:
        logging.info("No CSV files found to process.")
        return

    for csv_file in csv_files:
        try:
            target_file = processed_path / csv_file.name

            if target_file.exists():
                logging.warning(f"File will be overwritten: {target_file}")

            shutil.copy2(csv_file, target_file)
            logging.info(f"Copied: {csv_file.name} -> {processed_path}")

            if delete_original:
                csv_file.unlink()
                logging.info(f"Deleted original: {csv_file.name}")

        except Exception as e:
            logging.error(f"Error processing {csv_file.name}: {e}")
