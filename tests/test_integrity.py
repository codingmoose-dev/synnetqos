from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_core_artifacts_exist():
    required_paths = [
        "data/synthetic/synnetqos-dataset.csv",
        "results/generator/generator_config.json",
        "results/external_alignment/external_alignment_summary.csv",
        "results/ml_benchmark/ml_benchmark_summary.csv",
        "results/ml_benchmark/ml_leakage_audit.csv",
        "figures/ml_benchmark/ml_task_result_bars.pdf",
        "figures/supplementary/ml_precision_recall_curves.pdf",
    ]

    missing = [path for path in required_paths if not (ROOT / path).exists()]
    assert not missing, f"Missing expected artifact(s): {missing}"


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


def test_ml_leakage_audit_does_not_include_target_columns():
    audit = pd.read_csv(ROOT / "results/ml_benchmark/ml_leakage_audit.csv")

    target_columns = {"Dropped_Connection", "Drop_t_plus_1"}
    bad_rows = audit[
        audit["column"].isin(target_columns)
        & audit["status"].eq("included")
    ]

    assert bad_rows.empty, "Target columns were incorrectly included as ML features."


def test_context_only_task_excludes_direct_qos_outcomes():
    feature_sets = pd.read_csv(ROOT / "results/ml_benchmark/ml_feature_sets.csv")

    task_c = feature_sets[
        feature_sets["task_id"].eq("C_future_drop_context_only")
    ]

    forbidden_features = {
        "Signal_Strength_dBm",
        "Download_Speed_Mbps",
        "Upload_Speed_Mbps",
        "Latency_ms",
        "Jitter_ms",
        "Dropped_Connection",
        "Drop_t_plus_1",
    }

    bad_features = set(task_c["feature"]) & forbidden_features
    assert not bad_features, f"Task C includes leakage-prone feature(s): {sorted(bad_features)}"