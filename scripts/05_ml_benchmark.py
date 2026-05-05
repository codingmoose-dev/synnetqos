# Run leakage-aware ML benchmarks for the SynNetQoS synthetic dataset.
#
# Main outputs:
# - results/ml_benchmark/ml_benchmark_summary.csv
# - results/ml_benchmark/ml_benchmark_run_metrics.csv
# - results/ml_benchmark/ml_task_definitions.csv
# - results/ml_benchmark/ml_feature_sets.csv
# - results/ml_benchmark/ml_leakage_audit.csv
# - results/ml_benchmark/ml_split_summary.csv
# - results/ml_benchmark/ml_confusion_summary.csv
# - results/ml_benchmark/ml_feature_importance.csv
# - results/ml_benchmark/ml_reproducibility_metadata.json
# - figures/ml_benchmark/ml_task_result_bars.pdf
# - figures/supplementary/ml_precision_recall_curves.pdf

from __future__ import annotations

from pathlib import Path

from synnetqos.io import (
    path_exists,
    read_csv,
    save_json,
    save_plot,
    sha256_of_file,
    write_csv,
)
from synnetqos.ml import DEFAULT_SEEDS, run_ml_benchmark
from synnetqos.plotting import (
    plot_ml_precision_recall_curves,
    plot_ml_task_result_bars,
    setup_plot_style,
)


def main() -> None:
    setup_plot_style()

    data_path = Path("data/synthetic/synnetqos-dataset.csv")
    result_dir = Path("results/ml_benchmark")
    main_figure_dir = Path("figures/ml_benchmark")
    supplementary_figure_dir = Path("figures/supplementary")

    if not path_exists(data_path):
        raise FileNotFoundError(f"Missing SynNetQoS dataset: {data_path}")

    print("Loading SynNetQoS dataset for leakage-aware ML benchmarking...")
    syn_df = read_csv(data_path)
    dataset_hash = sha256_of_file(data_path)

    print("Running repeated session-wise ML benchmark...")
    benchmark = run_ml_benchmark(syn_df, seeds=DEFAULT_SEEDS)

    write_csv(benchmark["summary"], result_dir / "ml_benchmark_summary.csv")
    write_csv(benchmark["run_metrics"], result_dir / "ml_benchmark_run_metrics.csv")
    write_csv(benchmark["task_definitions"], result_dir / "ml_task_definitions.csv")
    write_csv(benchmark["feature_sets"], result_dir / "ml_feature_sets.csv")
    write_csv(benchmark["leakage_audit"], result_dir / "ml_leakage_audit.csv")
    write_csv(benchmark["split_summary"], result_dir / "ml_split_summary.csv")
    write_csv(benchmark["confusion_summary"], result_dir / "ml_confusion_summary.csv")

    feature_importance = benchmark["feature_importance"]

    if not feature_importance.empty:
        write_csv(feature_importance, result_dir / "ml_feature_importance.csv")

    print("Generating ML benchmark figures...")

    fig_bars = plot_ml_task_result_bars(
        benchmark["summary"],
        metric="average_precision",
    )
    save_plot(fig_bars, main_figure_dir / "ml_task_result_bars.pdf", dpi=300)

    fig_pr = plot_ml_precision_recall_curves(
        benchmark["curve_rows"],
        task_id="C_future_drop_context_only",
        title="Precision-Recall Curves: Leakage-Controlled Future-Drop Prediction",
    )
    save_plot(
        fig_pr,
        supplementary_figure_dir / "ml_precision_recall_curves.pdf",
        dpi=300,
    )

    metadata = {
        "dataset_path": str(data_path),
        "dataset_sha256": dataset_hash,
        "n_rows": int(syn_df.shape[0]),
        "n_columns": int(syn_df.shape[1]),
        "seeds": list(DEFAULT_SEEDS),
        "split_strategy": "Session-wise GroupShuffleSplit; 60/20/20 train/validation/test",
        "threshold_strategy": "Threshold selected on validation set by maximum F1; test set used only once for reporting.",
        "class_imbalance_strategy": "Class-weighted tree-based models where applicable; no SMOTE or row duplication.",
        "primary_interpretation": (
            "ML benchmark evaluates internal learnability under leakage-aware "
            "splitting; it does not prove real-world deployment accuracy."
        ),
    }

    save_json(metadata, result_dir / "ml_reproducibility_metadata.json")

    print("ML benchmark complete.")
    print(f"Results saved to: {result_dir}")
    print(f"Main figures saved to: {main_figure_dir}")
    print(f"Supplementary figures saved to: {supplementary_figure_dir}")

if __name__ == "__main__":
    main()