from __future__ import annotations

import numpy as np
import pandas as pd

from synnetqos.simulator_comparison import (
    SIMULATOR_SOURCE,
    SYNNETQOS_SOURCE,
    LOAD_BEARING_TRAFFIC_CLASS,
    LOW_RATE_TRAFFIC_CLASS,
    assign_offered_load_labels_from_series,
    build_synnetqos_simulator_subset,
    compare_simulator_kpis,
    metadata_from_cttc_filename,
    parse_cttc_nr_demo_text,
    simulator_comparison_interpretation_flags,
    simulator_comparison_verdict,
    simulator_kpi_summary,
    simulator_kpi_trend_summary,
)

SAMPLE_CTTC_NR_DEMO_OUTPUT = """
Flow 1 (7.0.0.2:49153 -> 1.0.0.2:9) proto UDP
  Tx Packets: 100
  Tx Bytes: 125200
  TxOffered: 1.669333 Mbps
  Rx Bytes: 125200
  Throughput: 1.669333 Mbps
  Mean delay: 0.553292 ms
  Mean jitter: 0.012300 ms
  Rx Packets: 100
Flow 2 (7.0.0.3:49154 -> 1.0.0.3:9) proto UDP
  Tx Packets: 100
  Tx Bytes: 125200
  TxOffered: 1.669333 Mbps
  Rx Bytes: 112680
  Throughput: 1.502400 Mbps
  Mean delay: 0.700000 ms
  Mean jitter: 0.020000 ms
  Rx Packets: 90

Mean flow throughput: 1.585866 Mbps
Mean flow delay: 0.626646 ms
"""


def test_parse_cttc_nr_demo_text_extracts_kpis() -> None:
    parsed = parse_cttc_nr_demo_text(
        SAMPLE_CTTC_NR_DEMO_OUTPUT,
        scenario_id="cttc_nr_demo_low_load_seed_1",
        raw_file="cttc_nr_demo_low_load_seed_1.txt",
        load_label="low",
        seed=1,
    )

    assert parsed.shape[0] == 2
    assert parsed["source"].unique().tolist() == [SIMULATOR_SOURCE]
    assert parsed["throughput_mbps"].notna().all()
    assert parsed["mean_delay_ms"].notna().all()
    assert parsed["mean_jitter_ms"].notna().all()

    packet_loss = parsed.loc[parsed["flow_id"] == 2, "packet_loss_ratio"].iloc[0]
    assert np.isclose(packet_loss, 0.10)


def test_metadata_from_cttc_filename() -> None:
    metadata = metadata_from_cttc_filename("cttc_nr_demo_high_load_seed_3.txt")

    assert metadata["scenario_id"] == "cttc_nr_demo_high_load_seed_3"
    assert metadata["load_label"] == "high"
    assert metadata["seed"] == 3


def test_assign_offered_load_labels_from_series_uses_tertiles() -> None:
    values = pd.Series([10.0, 20.0, 30.0, 100.0, 200.0, 300.0])
    labels = assign_offered_load_labels_from_series(values)

    assert labels.tolist().count("low") == 2
    assert labels.tolist().count("medium") == 2
    assert labels.tolist().count("high") == 2


def test_build_synnetqos_simulator_subset_keeps_controlled_5g_rows() -> None:
    df = pd.DataFrame(
        {
            "Network_Type": [
                "4G",
                "5G NSA",
                "5G NSA",
                "5G SA",
                "5G SA",
                "5G NSA",
                "5G SA",
            ],
            "Download_Speed_Mbps": [25.0, 80.0, 120.0, 180.0, 70.0, 60.0, 50.0],
            "Latency_ms": [40.0, 25.0, 15.0, 10.0, 30.0, 35.0, 45.0],
            "Jitter_ms": [8.0, 3.0, 2.0, 1.5, 4.0, 5.0, 6.0],
            "Packet_Loss_Rate": [2.0, 1.0, 0.5, 0.2, 1.5, 1.8, 2.2],
            "Congestion_Level": ["High", "Low", "Medium", "Medium", "High", "Low", "Low"],
            "Offered_Downlink_Mbps": [5.0, 10.0, 100.0, 300.0, 200.0, 150.0, 250.0],
            "Infrastructure_Profile": [
                "Nominal",
                "Nominal",
                "Nominal",
                "Nominal",
                "Nominal",
                "Moderately_Degraded",
                "Nominal",
            ],
            "Movement_Speed": [
                "Static",
                "Static",
                "Static",
                "Static",
                "Static",
                "Static",
                "Walking",
            ],
            "Signal_Strength_dBm": [-80.0, -75.0, -78.0, -76.0, -77.0, -74.0, -73.0],
        }
    )

    subset = build_synnetqos_simulator_subset(df)

    assert subset.shape[0] == 3
    assert subset["source"].unique().tolist() == [SYNNETQOS_SOURCE]
    assert set(subset["load_label"]) == {"low", "medium", "high"}
    assert subset["tx_offered_mbps"].notna().all()
    assert subset["packet_loss_ratio"].max() <= 1.0


def test_summary_comparison_and_flags_run() -> None:
    simulator = parse_cttc_nr_demo_text(
        SAMPLE_CTTC_NR_DEMO_OUTPUT,
        scenario_id="cttc_nr_demo_low_load_seed_1",
        raw_file="cttc_nr_demo_low_load_seed_1.txt",
        load_label="low",
        seed=1,
    )

    synnetqos = simulator.copy()
    synnetqos["source"] = SYNNETQOS_SOURCE
    synnetqos["traffic_class"] = "synthetic_session_kpi"
    synnetqos["scenario_id"] = "synnetqos_5g_simulator_comparable_subset"

    combined = pd.concat([simulator, synnetqos], ignore_index=True)

    summary = simulator_kpi_summary(combined)
    comparison = compare_simulator_kpis(combined)
    flags = simulator_comparison_interpretation_flags(comparison)

    assert not summary.empty
    assert not comparison.empty
    assert not flags.empty
    assert "median_ratio_synnetqos_over_simulator" in comparison.columns


def test_cttc_flow_classification_marks_only_load_bearing_flow() -> None:
    sample = """
Flow 1 (1.0.0.2:49153 -> 7.0.0.2:1234) proto UDP
  Tx Packets: 1
  Tx Bytes:   128
  TxOffered:  0.001707 Mbps
  Rx Bytes:   128
  Throughput: 0.001707 Mbps
  Mean delay:  0.675891 ms
  Mean jitter:  0.000000 ms
  Rx Packets: 1
Flow 2 (1.0.0.2:49154 -> 7.0.0.3:1234) proto UDP
  Tx Packets: 1
  Tx Bytes:   128
  TxOffered:  0.001707 Mbps
  Rx Bytes:   128
  Throughput: 0.001707 Mbps
  Mean delay:  0.671427 ms
  Mean jitter:  0.000000 ms
  Rx Packets: 1
Flow 3 (1.0.0.2:49155 -> 7.0.0.4:1235) proto UDP
  Tx Packets: 1200
  Tx Bytes:   1536000
  TxOffered:  20.480000 Mbps
  Rx Bytes:   1534720
  Throughput: 20.462933 Mbps
  Mean delay:  0.924287 ms
  Mean jitter:  0.001340 ms
  Rx Packets: 1199
"""

    parsed = parse_cttc_nr_demo_text(
        sample,
        scenario_id="cttc_nr_demo_low_load_seed_1",
        raw_file="cttc_nr_demo_low_load_seed_1.txt",
        load_label="low",
        seed=1,
    )

    assert parsed.shape[0] == 3

    class_by_flow = parsed.set_index("flow_id")["traffic_class"].to_dict()

    assert class_by_flow[1] == LOW_RATE_TRAFFIC_CLASS
    assert class_by_flow[2] == LOW_RATE_TRAFFIC_CLASS
    assert class_by_flow[3] == LOAD_BEARING_TRAFFIC_CLASS


def test_simulator_kpi_summary_uses_load_bearing_simulator_flow() -> None:
    sample = """
Flow 1 (1.0.0.2:49153 -> 7.0.0.2:1234) proto UDP
  Tx Packets: 1
  Tx Bytes:   128
  TxOffered:  0.001707 Mbps
  Rx Bytes:   128
  Throughput: 0.001707 Mbps
  Mean delay:  0.675891 ms
  Mean jitter:  0.000000 ms
  Rx Packets: 1
Flow 2 (1.0.0.2:49154 -> 7.0.0.3:1234) proto UDP
  Tx Packets: 1
  Tx Bytes:   128
  TxOffered:  0.001707 Mbps
  Rx Bytes:   128
  Throughput: 0.001707 Mbps
  Mean delay:  0.671427 ms
  Mean jitter:  0.000000 ms
  Rx Packets: 1
Flow 3 (1.0.0.2:49155 -> 7.0.0.4:1235) proto UDP
  Tx Packets: 1200
  Tx Bytes:   1536000
  TxOffered:  20.480000 Mbps
  Rx Bytes:   1534720
  Throughput: 20.462933 Mbps
  Mean delay:  0.924287 ms
  Mean jitter:  0.001340 ms
  Rx Packets: 1199
"""

    simulator = parse_cttc_nr_demo_text(
        sample,
        scenario_id="cttc_nr_demo_low_load_seed_1",
        raw_file="cttc_nr_demo_low_load_seed_1.txt",
        load_label="low",
        seed=1,
    )

    synnetqos = simulator.loc[
        simulator["traffic_class"] == LOAD_BEARING_TRAFFIC_CLASS
    ].copy()
    synnetqos["source"] = SYNNETQOS_SOURCE
    synnetqos["traffic_class"] = "synthetic_session_kpi"
    synnetqos["throughput_mbps"] = 22.0

    combined = pd.concat([simulator, synnetqos], ignore_index=True)
    summary = simulator_kpi_summary(combined)

    simulator_throughput = summary[
        (summary["source"] == SIMULATOR_SOURCE)
        & (summary["load_label"] == "low")
        & (summary["kpi"] == "throughput_mbps")
    ]["median"].iloc[0]

    assert np.isclose(simulator_throughput, 20.462933)


def test_assign_offered_load_labels_from_series_uses_tertiles() -> None:
    values = pd.Series([10.0, 20.0, 30.0, 100.0, 200.0, 300.0])
    labels = assign_offered_load_labels_from_series(values)

    assert labels.tolist().count("low") == 2
    assert labels.tolist().count("medium") == 2
    assert labels.tolist().count("high") == 2

def test_mixed_trends_are_ambiguous_not_matched() -> None:
    combined = pd.DataFrame(
        {
            "source": [
                SIMULATOR_SOURCE, SIMULATOR_SOURCE, SIMULATOR_SOURCE,
                SYNNETQOS_SOURCE, SYNNETQOS_SOURCE, SYNNETQOS_SOURCE,
            ],
            "scenario_id": ["sim", "sim", "sim", "syn", "syn", "syn"],
            "raw_file": ["", "", "", "", "", ""],
            "load_label": ["low", "medium", "high", "low", "medium", "high"],
            "seed": [1, 1, 1, np.nan, np.nan, np.nan],
            "flow_id": [1, 1, 1, np.nan, np.nan, np.nan],
            "protocol": ["UDP", "UDP", "UDP", "", "", ""],
            "endpoint": ["", "", "", "", "", ""],
            "traffic_class": [
                LOAD_BEARING_TRAFFIC_CLASS,
                LOAD_BEARING_TRAFFIC_CLASS,
                LOAD_BEARING_TRAFFIC_CLASS,
                "synthetic_session_kpi",
                "synthetic_session_kpi",
                "synthetic_session_kpi",
            ],
            "tx_packets": [100, 100, 100, np.nan, np.nan, np.nan],
            "rx_packets": [100, 100, 100, np.nan, np.nan, np.nan],
            "tx_bytes": [1, 1, 1, np.nan, np.nan, np.nan],
            "rx_bytes": [1, 1, 1, np.nan, np.nan, np.nan],
            "tx_offered_mbps": [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
            "throughput_mbps": [10.0, 20.0, 30.0, 12.0, 22.0, 32.0],
            "mean_delay_ms": [1.0, 0.5, 3.0, 10.0, 12.0, 11.0],
            "mean_jitter_ms": [0.1, 0.2, 0.15, 0.3, 0.25, 0.35],
            "packet_loss_ratio": [0.0, 0.0, 0.0, np.nan, np.nan, np.nan],
            "mean_flow_throughput_mbps": [np.nan] * 6,
            "mean_flow_delay_ms": [np.nan] * 6,
        }
    )

    trend = simulator_kpi_trend_summary(combined)
    comparison = trend[
        (trend["source"] == "trend_comparison")
        & (trend["kpi"] == "mean_delay_ms")
    ].iloc[0]

    assert comparison["trend_direction"] == "ambiguous"


def test_simulator_comparison_verdict_keeps_delay_diagnostic() -> None:
    combined = pd.DataFrame(
        {
            "source": [SIMULATOR_SOURCE, SYNNETQOS_SOURCE],
            "scenario_id": ["sim", "syn"],
            "raw_file": ["", ""],
            "load_label": ["low", "low"],
            "seed": [1, np.nan],
            "flow_id": [1, np.nan],
            "protocol": ["UDP", ""],
            "endpoint": ["", ""],
            "traffic_class": [LOAD_BEARING_TRAFFIC_CLASS, "synthetic_session_kpi"],
            "tx_packets": [100, np.nan],
            "rx_packets": [100, np.nan],
            "tx_bytes": [1, np.nan],
            "rx_bytes": [1, np.nan],
            "tx_offered_mbps": [1.0, np.nan],
            "throughput_mbps": [20.0, 21.0],
            "mean_delay_ms": [1.0, 50.0],
            "mean_jitter_ms": [0.01, 0.05],
            "packet_loss_ratio": [0.0, np.nan],
            "mean_flow_throughput_mbps": [np.nan, np.nan],
            "mean_flow_delay_ms": [np.nan, np.nan],
        }
    )

    comparison = compare_simulator_kpis(combined)
    flags = simulator_comparison_interpretation_flags(comparison)
    trend = simulator_kpi_trend_summary(combined)
    verdict = simulator_comparison_verdict(comparison, trend, flags)

    delay_row = verdict[verdict["component"] == "delay"].iloc[0]
    overall_row = verdict[verdict["component"] == "overall"].iloc[0]

    assert delay_row["status"] == "diagnostic_only"
    assert overall_row["status"] == "supplementary_simulator_reference"
