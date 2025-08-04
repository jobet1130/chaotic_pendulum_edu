import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from tqdm import tqdm


def load_config(config_path: str = "pendulum_config.json") -> dict:
    with open(config_path, "r") as file:
        return json.load(file)


def pendulum_ode(t, y, b, c):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -b * omega - np.sin(theta) + c * np.cos(t)
    return [dtheta_dt, domega_dt]


def simulate(config: dict):
    b = config["damping"]
    time_span = (0, config["total_time"])
    t_eval = np.arange(time_span[0], time_span[1], config["time_step"])

    theta0 = config["initial_angle"]
    omega0 = config["initial_velocity"]
    initial_conditions = [theta0, omega0]

    c_values = np.linspace(0.5, 1.5, config.get("batch_simulations", 10))

    result_dir = Path("data/raw")
    result_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    print("Running simulations...")
    for i, c in enumerate(tqdm(c_values, desc="Drive force loop")):
        sol = solve_ivp(
            pendulum_ode,
            time_span,
            initial_conditions,
            args=(b, c),
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-9,
        )

        df = pd.DataFrame(
            {"t": sol.t, "theta": sol.y[0], "omega": sol.y[1], "drive_force_c": c}
        )

        all_data.append(df)

        if (
            config.get("plot_sample", False)
            and i % config.get("plot_every_n", 100) == 0
        ):
            plt.figure(figsize=(10, 4))
            plt.plot(sol.t, sol.y[0], label=r"$\theta(t)$")
            plt.title(f"Driven Pendulum: c={c:.4f}")
            plt.xlabel("Time")
            plt.ylabel("Angle (rad)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plot_path = result_dir / f"plot_c_{c:.4f}.png"
            plt.savefig(plot_path)
            plt.close()

    combined_df = pd.concat(all_data, ignore_index=True)
    output_path = result_dir / "chaotic_pendulum_simulations.csv"
    combined_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "pendulum_config.json"
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    simulate(config)
    print("✅ Simulation complete. Combined CSV saved in `data/raw/`.")
