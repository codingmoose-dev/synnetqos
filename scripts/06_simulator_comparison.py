from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from synnetqos.io import read_csv, save_json, save_plot, write_csv
from synnetqos.plotting import setup_plot_style
from synnetqos.simulator_comparison import (
    LOAD_BEARING_TRAFFIC_CLASS,
    LOW_RATE_TRAFFIC_CLASS,
    SIMULATOR_SOURCE,
    build_synnetqos_simulator_subset,
    compare_simulator_kpis,
    ensure_simulator_schema,
    metadata_from_cttc_filename,
    parse_cttc_nr_demo_text,
    plot_simulator_delay_jitter_comparison,
    plot_simulator_throughput_comparison,
    simulator_comparison_interpretation_flags,
    simulator_feature_mapping_table,
    simulator_comparison_verdict,
    simulator_kpi_summary,
    simulator_kpi_trend_summary,
    simulator_packet_loss_summary,
)


SYNTHETIC_DATA_PATH = "data/synthetic/synnetqos-dataset.csv"

NS3_LENA_RAW_DIR = Path("data/external/ns3_lena/raw")
NS3_LENA_PROCESSED_DIR = Path("data/external/ns3_lena/processed")
SIMULATOR_COMPARISON_RESULTS_DIR = Path("results/simulator_comparison")
SIMULATOR_COMPARISON_FIGURES_DIR = Path("figures/simulator_comparison")
SUPPLEMENTARY_FIGURES_DIR = Path("figures/supplementary")

NS3_LENA_RUN_MANIFEST_PATH = NS3_LENA_RAW_DIR / "ns3_lena_run_manifest.csv"

NS3_LENA_KPIS_PATH = NS3_LENA_PROCESSED_DIR / "ns3_lena_kpis_normalized.csv"
SYNNETQOS_SIMULATOR_SUBSET_PATH = NS3_LENA_PROCESSED_DIR / "synnetqos_simulator_comparable_subset.csv"
SIMULATOR_COMPARISON_COMBINED_PATH = NS3_LENA_PROCESSED_DIR / "simulator_comparison_combined_schema.csv"

SIMULATOR_TRACE_MANIFEST_PATH = SIMULATOR_COMPARISON_RESULTS_DIR / "simulator_trace_manifest.json"
SIMULATOR_KPI_SUMMARY_PATH = SIMULATOR_COMPARISON_RESULTS_DIR / "simulator_kpi_summary.csv"
SIMULATOR_KPI_COMPARISON_PATH = SIMULATOR_COMPARISON_RESULTS_DIR / "simulator_kpi_comparison.csv"
SIMULATOR_INTERPRETATION_FLAGS_PATH = SIMULATOR_COMPARISON_RESULTS_DIR / "simulator_comparison_interpretation_flags.csv"
SIMULATOR_FEATURE_MAPPING_PATH = SIMULATOR_COMPARISON_RESULTS_DIR / "simulator_feature_mapping.csv"
SIMULATOR_TREND_SUMMARY_PATH = SIMULATOR_COMPARISON_RESULTS_DIR / "simulator_kpi_trend_summary.csv"
SIMULATOR_PACKET_LOSS_SUMMARY_PATH = SIMULATOR_COMPARISON_RESULTS_DIR / "simulator_packet_loss_summary.csv"
SIMULATOR_TRACE_QUALITY_FLAGS_PATH = SIMULATOR_COMPARISON_RESULTS_DIR / "simulator_trace_quality_flags.csv"
SIMULATOR_COMPARISON_VERDICT_PATH = SIMULATOR_COMPARISON_RESULTS_DIR / "simulator_comparison_verdict.csv"

EXPECTED_SIM_TIME = "30000ms"
EXPECTED_SEEDS_PER_LOAD = 20
EXPECTED_LOAD_LABELS = ("low", "medium", "high")
EXPECTED_RAW_FILE_COUNT = len(EXPECTED_LOAD_LABELS) * EXPECTED_SEEDS_PER_LOAD

SIMULATOR_THROUGHPUT_FIGURE_PATH = SIMULATOR_COMPARISON_FIGURES_DIR / "simulator_throughput_comparison.pdf"
SIMULATOR_DELAY_JITTER_FIGURE_PATH = SUPPLEMENTARY_FIGURES_DIR / "simulator_delay_jitter_comparison.pdf"


def main() -> None:
    print("Running KPI-level 5G-LENA/ns-3 simulator comparison...")
    setup_plot_style()

    raw_manifest = read_optional_manifest(NS3_LENA_RUN_MANIFEST_PATH)
    validate_publication_manifest(raw_manifest, NS3_LENA_RAW_DIR)
    raw_files = raw_files_from_manifest(raw_manifest, NS3_LENA_RAW_DIR)

    ns3_df = parse_raw_simulator_outputs(raw_files, raw_manifest)
    write_csv(ns3_df, NS3_LENA_KPIS_PATH)

    synthetic_df = read_csv(SYNTHETIC_DATA_PATH)
    synnetqos_subset = build_synnetqos_simulator_subset(synthetic_df)
    write_csv(synnetqos_subset, SYNNETQOS_SIMULATOR_SUBSET_PATH)

    combined_df = ensure_simulator_schema(pd.concat([ns3_df, synnetqos_subset], ignore_index=True))
    write_csv(combined_df, SIMULATOR_COMPARISON_COMBINED_PATH)

    kpi_summary = simulator_kpi_summary(combined_df)
    kpi_comparison = compare_simulator_kpis(combined_df)
    interpretation_flags = simulator_comparison_interpretation_flags(kpi_comparison)
    feature_mapping = simulator_feature_mapping_table()
    trend_summary = simulator_kpi_trend_summary(combined_df)
    packet_loss_summary = simulator_packet_loss_summary(combined_df)
    trace_quality_flags = simulator_trace_quality_flags(raw_manifest, ns3_df)
    comparison_verdict = simulator_comparison_verdict(
        kpi_comparison,
        trend_summary,
        interpretation_flags,
    )

    write_csv(kpi_summary, SIMULATOR_KPI_SUMMARY_PATH)
    write_csv(kpi_comparison, SIMULATOR_KPI_COMPARISON_PATH)
    write_csv(interpretation_flags, SIMULATOR_INTERPRETATION_FLAGS_PATH)
    write_csv(feature_mapping, SIMULATOR_FEATURE_MAPPING_PATH)
    write_csv(trend_summary, SIMULATOR_TREND_SUMMARY_PATH)
    write_csv(packet_loss_summary, SIMULATOR_PACKET_LOSS_SUMMARY_PATH)
    write_csv(trace_quality_flags, SIMULATOR_TRACE_QUALITY_FLAGS_PATH)
    write_csv(comparison_verdict, SIMULATOR_COMPARISON_VERDICT_PATH)

    save_json(
        build_trace_manifest(raw_files, raw_manifest, ns3_df),
        SIMULATOR_TRACE_MANIFEST_PATH,
    )

    save_plot(
        plot_simulator_throughput_comparison(combined_df),
        SIMULATOR_THROUGHPUT_FIGURE_PATH,
    )
    save_plot(
        plot_simulator_delay_jitter_comparison(combined_df),
        SIMULATOR_DELAY_JITTER_FIGURE_PATH,
    )

    print("Simulator comparison outputs saved.")
    print(f"Processed simulator KPIs: {NS3_LENA_KPIS_PATH}")
    print(f"Combined comparison schema: {SIMULATOR_COMPARISON_COMBINED_PATH}")
    print(f"Results directory: {SIMULATOR_COMPARISON_RESULTS_DIR}")
    print(f"Figures directory: {SIMULATOR_COMPARISON_FIGURES_DIR}")


def read_optional_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    manifest = read_csv(path)

    if "output_file" not in manifest.columns:
        raise KeyError(f"{path} is missing required column: output_file")

    return manifest



def validate_publication_manifest(raw_manifest: pd.DataFrame, raw_dir: Path) -> None:
    if raw_manifest.empty:
        raise FileNotFoundError(
            f"Missing required simulator run manifest: {NS3_LENA_RUN_MANIFEST_PATH}. "
            "Run simulators/ns3_lena/run_cttc_nr_demo_grid.sh before this workflow."
        )

    required = {"output_file", "load_label", "seed", "sim_time"}
    missing = required - set(raw_manifest.columns)
    if missing:
        raise KeyError(f"Simulator run manifest is missing required column(s): {sorted(missing)}")

    records = raw_manifest.copy()
    records["load_label"] = records["load_label"].astype(str).str.lower()
    records["seed"] = pd.to_numeric(records["seed"], errors="coerce").astype("Int64")
    records["sim_time"] = records["sim_time"].astype(str)

    if len(records) != EXPECTED_RAW_FILE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RAW_FILE_COUNT} simulator manifest records "
            f"({EXPECTED_SEEDS_PER_LOAD} seeds for each of {EXPECTED_LOAD_LABELS}), "
            f"but found {len(records)}. Rerun simulators/ns3_lena/run_cttc_nr_demo_grid.sh."
        )

    observed_loads = tuple(sorted(records["load_label"].dropna().unique()))
    if observed_loads != tuple(sorted(EXPECTED_LOAD_LABELS)):
        raise ValueError(f"Unexpected simulator load labels: {observed_loads}; expected {EXPECTED_LOAD_LABELS}.")

    observed_sim_times = set(records["sim_time"].dropna().unique())
    if observed_sim_times != {EXPECTED_SIM_TIME}:
        raise ValueError(
            f"Unexpected simulator sim_time values: {sorted(observed_sim_times)}; "
            f"expected only {EXPECTED_SIM_TIME}. Rerun the simulator grid."
        )

    for load_label, group in records.groupby("load_label"):
        seeds = sorted(int(seed) for seed in group["seed"].dropna().unique())
        expected_seeds = list(range(1, EXPECTED_SEEDS_PER_LOAD + 1))
        if seeds != expected_seeds:
            raise ValueError(
                f"Unexpected seed set for load label {load_label}: {seeds}; "
                f"expected {expected_seeds}. Rerun the simulator grid."
            )

    missing_files = [
        str(raw_dir / output_file)
        for output_file in records["output_file"].astype(str)
        if not (raw_dir / output_file).is_file() or (raw_dir / output_file).stat().st_size == 0
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing or empty simulator output file(s):\n" + "\n".join(f"- {path}" for path in missing_files)
        )


def raw_files_from_manifest(raw_manifest: pd.DataFrame, raw_dir: Path) -> list[Path]:
    return [raw_dir / output_file for output_file in raw_manifest["output_file"].astype(str).tolist()]

def parse_raw_simulator_outputs(raw_files: list[Path], raw_manifest: pd.DataFrame) -> pd.DataFrame:
    parsed_tables: list[pd.DataFrame] = []

    for raw_file in raw_files:
        metadata = metadata_from_cttc_filename(raw_file)

        if not raw_manifest.empty:
            matched = raw_manifest[raw_manifest["output_file"] == raw_file.name]

            if not matched.empty:
                manifest_row = matched.iloc[0].to_dict()
                metadata["scenario_id"] = manifest_row.get("scenario_id", metadata["scenario_id"])
                metadata["raw_file"] = manifest_row.get("output_file", metadata["raw_file"])
                metadata["load_label"] = manifest_row.get("load_label", metadata["load_label"])
                metadata["seed"] = manifest_row.get("seed", metadata["seed"])

        parsed = parse_cttc_nr_demo_text(
            raw_file.read_text(encoding="utf-8", errors="replace"),
            scenario_id=str(metadata["scenario_id"]),
            raw_file=str(metadata["raw_file"]),
            load_label=str(metadata["load_label"]),
            seed=metadata["seed"],
        )

        if parsed.empty:
            raise ValueError(f"No KPI rows were parsed from {raw_file}")

        parsed_tables.append(parsed)

    out = ensure_simulator_schema(pd.concat(parsed_tables, ignore_index=True))

    if out["throughput_mbps"].notna().sum() == 0:
        raise ValueError("No throughput values were parsed from the simulator traces.")

    return out


def build_trace_manifest(
    raw_files: list[Path],
    raw_manifest: pd.DataFrame,
    ns3_df: pd.DataFrame,
) -> dict[str, object]:
    parsed_rows_by_load = (
        ns3_df.groupby("load_label", dropna=False)
        .size()
        .rename("parsed_flow_rows")
        .reset_index()
        .to_dict(orient="records")
    )

    if raw_manifest.empty:
        raw_manifest_records = [{"output_file": path.name} for path in raw_files]
    else:
        raw_manifest_records = raw_manifest.to_dict(orient="records")

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_name": "controlled_5g_lena_ns3_kpi_level_simulator_comparison",
        "source": SIMULATOR_SOURCE,
        "raw_file_count": len(raw_files),
        "raw_files": [path.name for path in raw_files],
        "raw_manifest_available": not raw_manifest.empty,
        "raw_manifest_records": raw_manifest_records,
        "parsed_flow_rows": int(ns3_df.shape[0]),
        "parsed_rows_by_load_label": parsed_rows_by_load,
        "expected_raw_file_count": EXPECTED_RAW_FILE_COUNT,
        "expected_seeds_per_load": EXPECTED_SEEDS_PER_LOAD,
        "expected_sim_time": EXPECTED_SIM_TIME,
        "load_bearing_flow_rows": int((ns3_df["traffic_class"] == LOAD_BEARING_TRAFFIC_CLASS).sum()),
        "low_rate_flow_rows": int((ns3_df["traffic_class"] == LOW_RATE_TRAFFIC_CLASS).sum()),
        "scope_note": (
            "This is a KPI-level simulator-trace comparison using selected outputs "
            "from controlled 5G-LENA/ns-3 cttc-nr-demo runs. It is not validation, "
            "calibration, or full PHY/MAC equivalence."
        ),
    }


def simulator_trace_quality_flags(raw_manifest: pd.DataFrame, ns3_df: pd.DataFrame) -> pd.DataFrame:
    records = raw_manifest.copy()
    records["load_label"] = records["load_label"].astype(str).str.lower()
    records["seed"] = pd.to_numeric(records["seed"], errors="coerce")

    rows: list[dict[str, object]] = []

    rows.append(
        {
            "check": "raw_file_count",
            "observed": int(len(records)),
            "expected": EXPECTED_RAW_FILE_COUNT,
            "status": "pass" if len(records) == EXPECTED_RAW_FILE_COUNT else "fail",
            "note": "Number of simulator output files listed in the run manifest.",
        }
    )

    rows.append(
        {
            "check": "sim_time",
            "observed": ";".join(sorted(records["sim_time"].astype(str).unique())),
            "expected": EXPECTED_SIM_TIME,
            "status": "pass" if set(records["sim_time"].astype(str).unique()) == {EXPECTED_SIM_TIME} else "fail",
            "note": "Simulation duration used for each cttc-nr-demo trace.",
        }
    )

    rows.append(
        {
            "check": "load_bearing_flow_rows",
            "observed": int((ns3_df["traffic_class"] == LOAD_BEARING_TRAFFIC_CLASS).sum()),
            "expected": EXPECTED_RAW_FILE_COUNT,
            "status": "pass" if int((ns3_df["traffic_class"] == LOAD_BEARING_TRAFFIC_CLASS).sum()) == EXPECTED_RAW_FILE_COUNT else "review",
            "note": "One load-bearing flow is expected per simulator trace.",
        }
    )

    rows.append(
        {
            "check": "low_rate_flow_rows",
            "observed": int((ns3_df["traffic_class"] == LOW_RATE_TRAFFIC_CLASS).sum()),
            "expected": EXPECTED_RAW_FILE_COUNT * 2,
            "status": "pass" if int((ns3_df["traffic_class"] == LOW_RATE_TRAFFIC_CLASS).sum()) == EXPECTED_RAW_FILE_COUNT * 2 else "review",
            "note": "Two low-rate auxiliary flows are expected per cttc-nr-demo trace and are excluded from KPI comparison.",
        }
    )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
