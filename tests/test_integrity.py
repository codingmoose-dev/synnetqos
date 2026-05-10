from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_core_outputs_exist():
    required_paths = [
        "data/synthetic/synnetqos-dataset.csv",
        "results/generator/generator_config.json",
        "results/generator/drop_event_summary.csv",
        "results/external_alignment/external_alignment_summary.csv",
        "results/ml_benchmark/ml_benchmark_summary.csv",
        "results/ml_benchmark/ml_leakage_audit.csv",
        "results/ml_benchmark/ml_target_prevalence.csv",
        "figures/ml_benchmark/ml_task_result_bars.pdf",
        "figures/supplementary/ml_streaming_qoe_precision_recall_curves.pdf",
        "figures/supplementary/ml_future_drop_precision_recall_curves.pdf",
    ]

    missing = [path for path in required_paths if not (ROOT / path).exists()]
    assert not missing, f"Missing expected output(s): {missing}"


def test_dataset_has_core_columns():
    df = pd.read_csv(ROOT / "data/synthetic/synnetqos-dataset.csv", nrows=5)

    required_columns = {
        "Session_ID",
        "Timestamp",
        "Network_Type",
        "Deployment_Area",
        "Operator_Profile",
        "UE_Profile",
        "Signal_Strength_dBm",
        "Download_Speed_Mbps",
        "Upload_Speed_Mbps",
        "Latency_ms",
        "Jitter_ms",
        "Dropped_Connection",
    }

    missing = required_columns - set(df.columns)
    assert not missing, f"Missing dataset column(s): {sorted(missing)}"


def test_generator_metadata_matches_expected_dataset_shape():
    with (ROOT / "results/generator/generator_config.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["num_sessions"] > 0
    assert metadata["session_length"] > 0
    assert "dataset_sha256" in metadata


def test_ml_task_definitions_are_current():
    task_definitions = pd.read_csv(ROOT / "results/ml_benchmark/ml_task_definitions.csv")

    expected_task_ids = {
        "A_high_latency_radio_context",
        "B_downlink_shortfall_demand_context",
        "C_streaming_qoe_impairment_context",
        "D_future_drop_radio_context",
    }

    observed_task_ids = set(task_definitions["task_id"])
    missing = expected_task_ids - observed_task_ids
    unexpected = observed_task_ids - expected_task_ids

    assert not missing, f"Missing expected ML task(s): {sorted(missing)}"
    assert not unexpected, f"Unexpected ML task(s): {sorted(unexpected)}"


def test_ml_leakage_audit_does_not_include_target_columns():
    audit = pd.read_csv(ROOT / "results/ml_benchmark/ml_leakage_audit.csv")

    target_columns = {
        "Dropped_Connection",
        "Drop_t_plus_1",
        "High_Latency_200ms",
        "Downlink_Service_Shortfall",
        "Streaming_QoE_Impaired",
    }

    bad_rows = audit[
        audit["column"].isin(target_columns)
        & audit["status"].eq("included")
    ]

    assert bad_rows.empty, "Target columns were incorrectly included as ML features."


def test_primary_tasks_exclude_direct_qos_outcomes():
    feature_sets = pd.read_csv(ROOT / "results/ml_benchmark/ml_feature_sets.csv")

    primary_tasks = feature_sets[feature_sets["analysis_role"].eq("primary")]

    forbidden_features = {
        "Download_Speed_Mbps",
        "Upload_Speed_Mbps",
        "Latency_ms",
        "Jitter_ms",
        "Ping_ms",
        "Video_Quality",
        "Video_Quality_Label",
        "Dropped_Connection",
        "Drop_t_plus_1",
        "High_Latency_200ms",
        "Downlink_Service_Shortfall",
        "Streaming_QoE_Impaired",
        "Downlink_Efficiency",
    }

    bad_features = set(primary_tasks["feature"]) & forbidden_features
    assert not bad_features, f"Primary ML tasks include leakage-prone feature(s): {sorted(bad_features)}"


def test_future_drop_task_excludes_direct_qos_outcomes():
    feature_sets = pd.read_csv(ROOT / "results/ml_benchmark/ml_feature_sets.csv")

    future_drop_task = feature_sets[
        feature_sets["task_id"].eq("D_future_drop_radio_context")
    ]

    forbidden_features = {
        "Download_Speed_Mbps",
        "Upload_Speed_Mbps",
        "Latency_ms",
        "Jitter_ms",
        "Ping_ms",
        "Video_Quality",
        "Video_Quality_Label",
        "Dropped_Connection",
        "Drop_t_plus_1",
        "High_Latency_200ms",
        "Downlink_Service_Shortfall",
        "Streaming_QoE_Impaired",
        "Downlink_Efficiency",
    }

    bad_features = set(future_drop_task["feature"]) & forbidden_features
    assert not bad_features, f"Future-drop task includes leakage-prone feature(s): {sorted(bad_features)}"


def test_drop_event_rate_is_benchmark_usable():
    with (ROOT / "results/generator/generator_config.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    audit = pd.read_csv(ROOT / "results/generator/drop_event_summary.csv")
    row = audit[
        audit["summary_type"].eq("overall_current_drop")
        & audit["category"].eq("all_rows")
    ]

    assert not row.empty, "Missing overall dropped-connection summary row."

    drop_rate = float(row["positive_rate"].iloc[0])
    lower = float(metadata.get("drop_rate_min", 0.01))
    upper = float(metadata.get("drop_rate_max", 0.035))

    assert lower <= drop_rate <= upper, (
        f"Dropped-connection rate {drop_rate:.4f} outside configured benchmark range "
        f"[{lower:.4f}, {upper:.4f}]."
    )
