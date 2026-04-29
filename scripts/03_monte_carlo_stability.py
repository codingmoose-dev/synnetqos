from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from synnetqos.generator import generate_full_dataset
from synnetqos.io import save_plot, write_csv

# Executes multiple passes of the dataset generator to ensure statistical convergence and stability across different random seeds.
def run_monte_carlo_stability(n_runs: int = 20, sessions_per_run: int = 5000, out_dir: str | Path = "figures/supplementary") -> None:
    print("--- Running Monte Carlo Generator Stability Check ---")
    run_stats: list[dict[str, float]] = []

    for i in range(n_runs):
        print(f"  Starting Run {i+1}/{n_runs}...")
        df_run: pd.DataFrame = generate_full_dataset(run_id=i, num_sessions=sessions_per_run)
        
        run_stats.append({
            "Latency (ms)": float(df_run["Latency_ms"].mean()),
            "Drop Rate (%)": float(df_run["Dropped_Connection"].mean() * 100),
            "Download Speed (Mbps)": float(df_run["Download_Speed_Mbps"].mean())
        })

    stats_df = pd.DataFrame(run_stats)
    output_path = Path(out_dir) / "monte_carlo_stability.pdf"
    fig, ax = plt.subplots()
    sns.boxplot(data=stats_df, ax=ax)
    ax.set_title(f"Generator Metrics Across {n_runs} Independent Runs")
    ax.set_ylabel("Mean Value per Full Dataset")
    save_plot(fig, output_path)

    # Calculate Coefficient of Variation (CV)
    summary = stats_df.describe()
    summary.loc['CV'] = summary.loc['std'] / summary.loc['mean'].abs()

    write_csv(stats_df, "results/generator/monte_carlo_run_metrics.csv")
    write_csv(summary.reset_index().rename(columns={"index": "Statistic"}), "results/generator/monte_carlo_stability_summary.csv")
    
    print("\n--- Monte Carlo Generator Stability Summary ---")
    print(summary.round(4))
    continuous_metrics = ["Latency (ms)", "Download Speed (Mbps)"]
    continuous_stable = (summary.loc["CV", continuous_metrics] < 0.05).all()

    drop_rate_std_pp = summary.loc["std", "Drop Rate (%)"]
    drop_rate_stable = drop_rate_std_pp < 0.5

    if continuous_stable and drop_rate_stable:
        print("\nResult: Continuous metrics show low CV, and the dropped-connection rate remains stable in absolute percentage-point terms.")
    else:
        print("\nResult: One or more metrics requires review. Check both CV and absolute standard deviation before interpreting stability.")

def main() -> None:
    # Encapsulating execution logic
    run_monte_carlo_stability()

if __name__ == "__main__":
    main()
    