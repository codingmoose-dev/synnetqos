from __future__ import annotations

from pathlib import Path

import pandas as pd

from synnetqos.generator import generate_full_dataset
from synnetqos.io import save_plot, write_csv
from synnetqos.plotting import plot_monte_carlo_stability, setup_plot_style


def run_monte_carlo_stability(
    n_runs: int = 20,
    sessions_per_run: int = 5000,
    session_length: int = 10,
    out_dir: str | Path = "figures/supplementary",
) -> None:
    print("--- Running Monte Carlo Generator Stability Check ---")
    setup_plot_style()

    metric_columns = [
        "Latency (ms)",
        "Drop Rate (%)",
        "Download Speed (Mbps)",
    ]

    plotted_metric_columns = [
        "Latency (ms)",
        "Download Speed (Mbps)",
    ]

    run_stats: list[dict[str, float | int]] = []

    for run_id in range(n_runs):
        print(f"  Starting Run {run_id + 1}/{n_runs}...")

        df_run: pd.DataFrame = generate_full_dataset(
            run_id=run_id,
            num_sessions=sessions_per_run,
            session_length=session_length,
        )

        run_stats.append(
            {
                "Run_ID": run_id,
                "Generator_Seed": run_id,
                "Num_Sessions": sessions_per_run,
                "Session_Length": session_length,
                "Latency (ms)": float(df_run["Latency_ms"].mean()),
                "Drop Rate (%)": float(df_run["Dropped_Connection"].mean() * 100),
                "Download Speed (Mbps)": float(df_run["Download_Speed_Mbps"].mean()),
            }
        )

    stats_df = pd.DataFrame(run_stats)
    metric_df = stats_df.loc[:, metric_columns]
    plot_df = stats_df.loc[:, plotted_metric_columns]

    output_path = Path(out_dir) / "monte_carlo_stability.pdf"

    fig = plot_monte_carlo_stability(
        plot_df,
        title="Monte Carlo Stability across Independent Generator Runs",
    )
    save_plot(fig, output_path)

    summary = metric_df.describe()
    summary.loc["CV"] = summary.loc["std"] / summary.loc["mean"].abs()

    write_csv(stats_df, "results/generator/monte_carlo_run_metrics.csv")
    write_csv(
        summary.reset_index().rename(columns={"index": "Statistic"}),
        "results/generator/monte_carlo_stability_summary.csv",
    )

    print("\n--- Monte Carlo Generator Stability Summary ---")
    print(summary.round(4))

    continuous_metrics = ["Latency (ms)", "Download Speed (Mbps)"]
    continuous_stable = (summary.loc["CV", continuous_metrics] < 0.05).all()

    drop_rate_std_pp = summary.loc["std", "Drop Rate (%)"]
    drop_rate_stable = drop_rate_std_pp < 0.5

    if continuous_stable and drop_rate_stable:
        print(
            "\nResult: Continuous metrics show low CV, and the dropped-connection "
            "rate remains stable in absolute percentage-point terms."
        )
    else:
        print(
            "\nResult: One or more metrics requires review. Check both CV and "
            "absolute standard deviation before interpreting stability."
        )


def main() -> None:
    run_monte_carlo_stability()


if __name__ == "__main__":
    main()