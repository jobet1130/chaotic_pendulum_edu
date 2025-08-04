import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import pairwise_distances

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


class ChaosLabeler:
    """Class for labeling chaotic pendulum data.

    This class provides methods to analyze pendulum simulation data and label it as
    chaotic or non-chaotic based on various metrics like Lyapunov exponents,
    correlation dimension, phase space analysis, and other chaos detection methods.
    """

    def __init__(self, config_path: str = "pendulum_config.json"):
        """Initialize the ChaosLabeler with configuration.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.labeled_data = None

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
        """Load pendulum simulation data.

        Args:
            data_path (Union[str, Path]): Path to the data file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logging.info(f"Loading data from {data_path}")
        print(f"📊 Loading data from {data_path}")

        df = pd.read_csv(data_path)
        return df

    def calculate_lyapunov_exponent(
        self,
        time_series: np.ndarray,
        embedding_dim: Optional[int] = None,
        time_delay: Optional[int] = None,
    ) -> float:
        """Calculate the largest Lyapunov exponent for a time series.

        The Lyapunov exponent measures the rate of separation of infinitesimally close
        trajectories in phase space, which is a key indicator of chaos.

        Args:
            time_series (np.ndarray): Time series data.
            embedding_dim (Optional[int]): Embedding dimension.
            time_delay (Optional[int]): Time delay.

        Returns:
            float: Estimated largest Lyapunov exponent.
        """
        if embedding_dim is None:
            embedding_dim = self.config["chaos_detection"]["embedding_dimension"]
        if time_delay is None:
            time_delay = self.config["chaos_detection"]["time_delay"]

        # Implement Rosenstein's algorithm for Lyapunov exponent calculation
        # This is a simplified version for educational purposes

        # Phase space reconstruction
        N = len(time_series)
        if N < (embedding_dim - 1) * time_delay + 100:
            logging.warning("Time series too short for reliable Lyapunov estimation")
            return 0.0

        # Create embedded vectors
        vectors = []
        for i in range(N - (embedding_dim - 1) * time_delay):
            vector = [time_series[i + j * time_delay] for j in range(embedding_dim)]
            vectors.append(vector)

        vectors = np.array(vectors)

        # Calculate pairwise distances
        distances = pairwise_distances(vectors)
        np.fill_diagonal(distances, np.inf)  # Exclude self-comparisons

        # Find nearest neighbors
        nearest_indices = np.argmin(distances, axis=1)

        # Track divergence over time
        max_steps = min(20, N - (embedding_dim - 1) * time_delay - 1)
        divergence = np.zeros(max_steps)

        for i in range(max_steps):
            # Calculate distances between points and their evolved neighbors
            valid_indices = np.arange(len(vectors) - i)
            evolved_indices = nearest_indices[valid_indices] + i
            mask = evolved_indices < len(vectors)

            if np.sum(mask) == 0:
                break

            valid_indices = valid_indices[mask]
            evolved_indices = evolved_indices[mask]

            # Calculate logarithm of distances
            d_i = np.linalg.norm(
                vectors[valid_indices + i] - vectors[evolved_indices], axis=1
            )
            d_i = d_i[d_i > 0]  # Avoid log(0)

            if len(d_i) > 0:
                divergence[i] = np.mean(np.log(d_i))

        # Estimate Lyapunov exponent from the slope of the divergence curve
        x = np.arange(1, len(divergence) + 1)
        slope, _, _, _, _ = stats.linregress(x, divergence)

        return slope

    def detect_chaos(
        self,
        df: pd.DataFrame,
        column: str = "num__theta",
        use_correlation_dim: bool = True,
    ) -> pd.DataFrame:
        """Detect chaos in the pendulum data.

        Args:
            df (pd.DataFrame): Pendulum simulation data.
            column (str): Column name to analyze for chaos.
            use_correlation_dim (bool): Whether to use correlation dimension as an 
                additional indicator.

        Returns:
            pd.DataFrame: DataFrame with chaos labels.
        """
        logging.info(f"Detecting chaos in {column} column")
        print(f"🔍 Detecting chaos in {column} column...")

        # Group by drive force parameter
        grouped = df.groupby("num__drive_force_c")

        results = []
        for drive_force, group in grouped:
            time_series = group[column].values

            # Calculate Lyapunov exponent
            lyapunov = self.calculate_lyapunov_exponent(time_series)

            # Calculate correlation dimension if requested
            corr_dim = 0.0
            if use_correlation_dim:
                corr_dim = self.calculate_correlation_dimension(time_series)
                logging.info(
                    f"Correlation dimension for drive force {drive_force:.4f}: "
                    f"{corr_dim:.4f}"
                )

            # Determine if chaotic based on Lyapunov exponent and correlation dimension
            # Use a lower threshold than in config for better sensitivity
            lyapunov_threshold = (
                self.config["chaos_detection"].get("threshold", 0.1) * 0.5
            )
            corr_dim_threshold = (
                1.5  # Chaotic systems often have correlation dimension > 1.5
            )

            # System is chaotic if either indicator suggests chaos
            is_chaotic_lyapunov = lyapunov > lyapunov_threshold
            is_chaotic_corr_dim = (
                corr_dim > corr_dim_threshold if use_correlation_dim else False
            )
            is_chaotic = is_chaotic_lyapunov or is_chaotic_corr_dim

            results.append(
                {
                    "drive_force_c": drive_force,
                    "lyapunov_exponent": lyapunov,
                    "correlation_dimension": corr_dim if use_correlation_dim else None,
                    "is_chaotic": is_chaotic,
                    "is_chaotic_lyapunov": is_chaotic_lyapunov,
                    "is_chaotic_corr_dim": (
                        is_chaotic_corr_dim if use_correlation_dim else None
                    ),
                }
            )

            logging.info(
                f"Drive force {drive_force:.4f}: Lyapunov={lyapunov:.6f}, "
                f"Corr. Dim={corr_dim:.4f}, Chaotic={is_chaotic}"
            )

        results_df = pd.DataFrame(results)
        print(
            f"✅ Chaos detection complete. Found "
            f"{results_df['is_chaotic'].sum()} chaotic cases."
        )

        # Merge results with original data
        labeled_df = df.merge(
            results_df[["drive_force_c", "is_chaotic"]],
            left_on="num__drive_force_c",
            right_on="drive_force_c",
            how="left",
        )

        self.labeled_data = labeled_df
        return labeled_df

    def calculate_correlation_dimension(
        self,
        time_series: np.ndarray,
        embedding_dim: Optional[int] = None,
        time_delay: Optional[int] = None,
        max_radius: float = 10.0,
        n_points: int = 20,
    ) -> float:
        """Calculate the correlation dimension for a time series.

        The correlation dimension is a measure of the dimensionality of the space
        occupied by a set of points, and is often used to characterize chaotic 
        attractors.

        Args:
            time_series (np.ndarray): Time series data.
            embedding_dim (Optional[int]): Embedding dimension.
            time_delay (Optional[int]): Time delay.
            max_radius (float): Maximum radius for correlation sum calculation.
            n_points (int): Number of points for radius values.

        Returns:
            float: Estimated correlation dimension.
        """
        if embedding_dim is None:
            embedding_dim = self.config["chaos_detection"]["embedding_dimension"]
        if time_delay is None:
            time_delay = self.config["chaos_detection"]["time_delay"]

        # Phase space reconstruction
        N = len(time_series)
        if N < (embedding_dim - 1) * time_delay + 100:
            logging.warning(
                "Time series too short for reliable correlation dimension estimation"
            )
            return 0.0

        # Create embedded vectors
        vectors = []
        for i in range(N - (embedding_dim - 1) * time_delay):
            vector = [time_series[i + j * time_delay] for j in range(embedding_dim)]
            vectors.append(vector)

        vectors = np.array(vectors)

        # Calculate pairwise distances
        distances = pairwise_distances(vectors)
        np.fill_diagonal(distances, np.inf)  # Exclude self-comparisons

        # Calculate correlation sum for different radii
        radii = np.logspace(-2, np.log10(max_radius), n_points)
        correlation_sum = np.zeros(len(radii))

        for i, r in enumerate(radii):
            correlation_sum[i] = np.sum(distances < r) / (N * (N - 1))

        # Calculate correlation dimension from the slope of log(C(r)) vs log(r)
        valid_indices = correlation_sum > 0
        if np.sum(valid_indices) < 3:
            logging.warning(
                "Not enough valid points for correlation dimension estimation"
            )
            return 0.0

        log_r = np.log(radii[valid_indices])
        log_c = np.log(correlation_sum[valid_indices])

        slope, _, _, _, _ = stats.linregress(log_r, log_c)

        return slope

    def plot_phase_space(
        self,
        df: pd.DataFrame,
        drive_forces: Optional[List[float]] = None,
        save_dir: Optional[Union[str, Path]] = None,
        plot_3d: bool = True,
    ) -> None:
        """Plot phase space diagrams for selected drive forces.

        Args:
            df (pd.DataFrame): Labeled pendulum data.
            drive_forces (Optional[List[float]]): List of drive forces to plot.
            save_dir (Optional[Union[str, Path]]): Directory to save plots.
            plot_3d (bool): Whether to create 3D phase space plots.
        """
        if drive_forces is None:
            # Select a few drive forces to plot
            if "is_chaotic" in df.columns and not df.empty:
                # Get some chaotic and non-chaotic examples
                chaotic = df[df["is_chaotic"]]["num__drive_force_c"].unique() if "num__drive_force_c" in df.columns else []
                non_chaotic = df[~df["is_chaotic"]][
                    "num__drive_force_c"
                ].unique() if "num__drive_force_c" in df.columns else []

                chaotic_sample = np.random.choice(
                    chaotic, min(2, len(chaotic)), replace=False
                )
                non_chaotic_sample = np.random.choice(
                    non_chaotic, min(2, len(non_chaotic)), replace=False
                )

                drive_forces = np.concatenate([chaotic_sample, non_chaotic_sample])
            else:
                # Just pick some random drive forces
                drive_forces = np.random.choice(
                    df["num__drive_force_c"].unique(), 4, replace=False
                )

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"📈 Plotting phase space diagrams for {len(drive_forces)} drive forces..."
        )

        for drive_force in drive_forces:
            subset = df[df["num__drive_force_c"] == drive_force]

            is_chaotic = "Unknown"
            if "is_chaotic" in subset.columns:
                is_chaotic = "Chaotic" if subset["is_chaotic"].iloc[0] else "Regular"

            # 2D Phase Space Plot
            plt.figure(figsize=(10, 8))
            plt.plot(subset["num__theta"], subset["num__omega"], "b-", alpha=0.5)
            plt.scatter(
                subset["num__theta"].iloc[0],
                subset["num__omega"].iloc[0],
                color="green",
                s=100,
                label="Start",
            )

            plt.title(f"Phase Space: Drive Force c={drive_force:.4f} ({is_chaotic})")
            plt.xlabel("Angle (θ)")
            plt.ylabel("Angular Velocity (ω)")
            plt.grid(True, alpha=0.3)
            plt.legend()

            if save_dir is not None:
                filename = f"phase_space_c_{drive_force:.4f}.png"
                plt.savefig(save_dir / filename, dpi=300, bbox_inches="tight")
                logging.info(f"Saved phase space plot to {save_dir / filename}")
            else:
                plt.show()

            plt.close()

            # 3D Phase Space Plot (if requested)
            if plot_3d and len(subset) > 100:

                # Create time-delayed coordinates for 3D plot
                delay = self.config["chaos_detection"]["time_delay"]
                theta = subset["num__theta"].values

                # Create time-delayed versions of theta for 3D visualization
                theta_t = theta[: -2 * delay]
                theta_t_plus_delay = theta[delay:-delay]
                theta_t_plus_2delay = theta[2 * delay :]

                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection="3d")

                # Plot 3D trajectory with color gradient to show time progression
                points = ax.scatter(
                    theta_t,
                    theta_t_plus_delay,
                    theta_t_plus_2delay,
                    c=np.arange(len(theta_t)),
                    cmap="viridis",
                    s=10,
                    alpha=0.8,
                )

                # Add colorbar to show time progression
                cbar = plt.colorbar(points, ax=ax, pad=0.1)
                cbar.set_label("Time Progression")

                ax.set_title(
                    f"3D Phase Space: Drive Force c={drive_force:.4f} ({is_chaotic})"
                )
                ax.set_xlabel(r"$\theta(t)$")
                ax.set_ylabel(r"$\theta(t+\tau)$")
                ax.set_zlabel(r"$\theta(t+2\tau)$")

                if save_dir is not None:
                    filename = f"phase_space_3d_c_{drive_force:.4f}.png"
                    plt.savefig(save_dir / filename, dpi=300, bbox_inches="tight")
                    logging.info(f"Saved 3D phase space plot to {save_dir / filename}")
                else:
                    plt.show()

                plt.close()

    def save_labeled_data(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save labeled data to CSV file.

        Args:
            output_path (Optional[Union[str, Path]]): Path to save labeled data.
        """
        if self.labeled_data is None:
            logging.warning("No labeled data to save")
            return

        if output_path is None:
            output_path = Path("data/labeled/chaotic_pendulum_labeled.csv")
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.labeled_data.to_csv(output_path, index=False)
        logging.info(f"Labeled data saved to {output_path}")
        print(f"💾 Labeled data saved to {output_path}")


def main():
    """Main function to run the chaos labeling process."""
    import argparse

    parser = argparse.ArgumentParser(description="Label chaotic pendulum data")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/chaotic_pendulum_simulations.csv",
        help="Path to the pendulum simulation data",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pendulum_config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/labeled/chaotic_pendulum_labeled.csv",
        help="Path to save labeled data",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="data/plots_labeled",
        help="Directory to save phase space plots",
    )
    parser.add_argument(
        "--use-corr-dim",
        action="store_true",
        help="Use correlation dimension for chaos detection",
    )
    parser.add_argument(
        "--no-3d-plots", action="store_true", help="Disable 3D phase space plots"
    )
    parser.add_argument(
        "--column",
        type=str,
        default="num__theta",
        help="Column to analyze for chaos detection",
    )

    args = parser.parse_args()

    labeler = ChaosLabeler(args.config)
    df = labeler.load_data(args.data)
    labeled_df = labeler.detect_chaos(
        df, column=args.column, use_correlation_dim=args.use_corr_dim
    )
    labeler.save_labeled_data(args.output)
    labeler.plot_phase_space(
        labeled_df, save_dir=args.plot_dir, plot_3d=not args.no_3d_plots
    )

    # Print summary of results
    if "is_chaotic" in labeled_df.columns:
        # Get unique drive forces and their chaotic status
        unique_forces = labeled_df[
            ["num__drive_force_c", "is_chaotic"]
        ].drop_duplicates()
        chaotic_count = unique_forces["is_chaotic"].sum()
        total_count = len(unique_forces)
        print(
            f"📊 Summary: {chaotic_count} out of {total_count} drive forces "
            f"exhibit chaotic behavior."
        )

    print("✨ Chaos labeling complete!")


if __name__ == "__main__":
    main()
