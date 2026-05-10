from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SIMULATOR_SOURCE = "5G-LENA/ns-3"
SYNNETQOS_SOURCE = "SynNetQoS"
LOAD_BEARING_TRAFFIC_CLASS = "load_bearing_flow"
LOW_RATE_TRAFFIC_CLASS = "low_rate_flow"
SYNNETQOS_TRAFFIC_CLASS = "synthetic_session_kpi"
LOAD_BEARING_TX_OFFERED_MBPS_MIN = 1.0

KPI_COLUMNS: tuple[str, ...] = (
    "throughput_mbps",
    "mean_delay_ms",
    "mean_jitter_ms",
    "packet_loss_ratio",
)

CROSS_SOURCE_KPI_COLUMNS: tuple[str, ...] = (
    "throughput_mbps",
    "mean_delay_ms",
    "mean_jitter_ms",
)

LOAD_LABEL_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "unassigned": 3}

SIMULATOR_COLUMNS: tuple[str, ...] = (
    "source",
    "scenario_id",
    "raw_file",
    "load_label",
    "seed",
    "flow_id",
    "protocol",
    "endpoint",
    "traffic_class",
    "tx_packets",
    "rx_packets",
    "tx_bytes",
    "rx_bytes",
    "tx_offered_mbps",
    "throughput_mbps",
    "mean_delay_ms",
    "mean_jitter_ms",
    "packet_loss_ratio",
    "mean_flow_throughput_mbps",
    "mean_flow_delay_ms",
)

CTTC_DEMO_FILENAME_PATTERN = re.compile(
    r"cttc_nr_demo_(?P<load_label>low|medium|high)_load_seed_(?P<seed>\d+)",
    re.IGNORECASE,
)

FLOW_HEADER_PATTERN = re.compile(
    r"^Flow\s+(?P<flow_id>\d+)\s+\((?P<endpoint>.*?)\)\s+proto\s+(?P<protocol>\S+)",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def require_columns(df: pd.DataFrame, required: Sequence[str], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{dataset_name} is missing required columns: {missing}")


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def parse_number(value: object) -> float:
    match = NUMBER_PATTERN.search(str(value))
    if match is None:
        return np.nan

    return float(match.group(0))


def metadata_from_cttc_filename(path: str | Path) -> dict[str, object]:
    filename = Path(path).name
    scenario_id = Path(path).stem
    match = CTTC_DEMO_FILENAME_PATTERN.search(scenario_id)

    metadata: dict[str, object] = {
        "scenario_id": scenario_id,
        "raw_file": filename,
        "load_label": "unassigned",
        "seed": np.nan,
    }

    if match:
        metadata["load_label"] = match.group("load_label").lower()
        metadata["seed"] = int(match.group("seed"))

    return metadata


def parse_cttc_nr_demo_text(
    text: str,
    scenario_id: str,
    raw_file: str,
    load_label: str,
    seed: int | float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_row: dict[str, object] | None = None

    mean_flow_throughput_mbps = np.nan
    mean_flow_delay_ms = np.nan

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        lower_line = line.lower()

        if lower_line.startswith("mean flow throughput"):
            mean_flow_throughput_mbps = parse_number(line)
            continue

        if lower_line.startswith("mean flow delay"):
            mean_flow_delay_ms = parse_number(line)
            continue

        flow_match = FLOW_HEADER_PATTERN.match(line)

        if flow_match:
            if current_row is not None:
                rows.append(current_row)

            current_row = {
                "source": SIMULATOR_SOURCE,
                "scenario_id": scenario_id,
                "raw_file": raw_file,
                "load_label": str(load_label).lower(),
                "seed": seed,
                "flow_id": int(flow_match.group("flow_id")),
                "protocol": flow_match.group("protocol"),
                "endpoint": flow_match.group("endpoint"),
                "traffic_class": "",
                "tx_packets": np.nan,
                "rx_packets": np.nan,
                "tx_bytes": np.nan,
                "rx_bytes": np.nan,
                "tx_offered_mbps": np.nan,
                "throughput_mbps": np.nan,
                "mean_delay_ms": np.nan,
                "mean_jitter_ms": np.nan,
            }
            continue

        if current_row is None:
            continue

        if lower_line.startswith("tx packets"):
            current_row["tx_packets"] = parse_number(line)
        elif lower_line.startswith("rx packets"):
            current_row["rx_packets"] = parse_number(line)
        elif lower_line.startswith("tx bytes"):
            current_row["tx_bytes"] = parse_number(line)
        elif lower_line.startswith("rx bytes"):
            current_row["rx_bytes"] = parse_number(line)
        elif lower_line.startswith("txoffered"):
            current_row["tx_offered_mbps"] = parse_number(line)
        elif lower_line.startswith("throughput"):
            current_row["throughput_mbps"] = parse_number(line)
        elif lower_line.startswith("mean delay"):
            current_row["mean_delay_ms"] = parse_number(line)
        elif lower_line.startswith("mean jitter"):
            current_row["mean_jitter_ms"] = parse_number(line)

    if current_row is not None:
        rows.append(current_row)

    out = pd.DataFrame(rows)

    if out.empty:
        return ensure_simulator_schema(out)
    
    out["mean_flow_throughput_mbps"] = mean_flow_throughput_mbps
    out["mean_flow_delay_ms"] = mean_flow_delay_ms
    out["packet_loss_ratio"] = packet_loss_ratio(out["tx_packets"], out["rx_packets"])

    out = ensure_simulator_schema(out)
    out["traffic_class"] = assign_simulator_traffic_class(out)

    return ensure_simulator_schema(out)


def ensure_simulator_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in SIMULATOR_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    numeric_columns = [
        "seed",
        "flow_id",
        "tx_packets",
        "rx_packets",
        "tx_bytes",
        "rx_bytes",
        "tx_offered_mbps",
        "throughput_mbps",
        "mean_delay_ms",
        "mean_jitter_ms",
        "packet_loss_ratio",
        "mean_flow_throughput_mbps",
        "mean_flow_delay_ms",
    ]

    for col in numeric_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["source"] = out["source"].fillna(SIMULATOR_SOURCE)
    out["load_label"] = out["load_label"].fillna("unassigned").astype(str).str.lower()

    return out.loc[:, list(SIMULATOR_COLUMNS)]


def packet_loss_ratio(tx_packets: pd.Series, rx_packets: pd.Series) -> pd.Series:
    tx = pd.to_numeric(tx_packets, errors="coerce")
    rx = pd.to_numeric(rx_packets, errors="coerce")
    ratio = (tx - rx) / tx
    ratio = ratio.where(tx > 0, np.nan)

    return ratio.clip(lower=0.0, upper=1.0)


def assign_simulator_traffic_class(df: pd.DataFrame) -> pd.Series:
    require_columns(
        df,
        ["source", "tx_offered_mbps", "tx_packets"],
        "simulator traffic-class assignment",
    )

    offered = clean_numeric(df["tx_offered_mbps"])
    packets = clean_numeric(df["tx_packets"])

    load_bearing = (
        df["source"].eq(SIMULATOR_SOURCE)
        & (
            (offered >= LOAD_BEARING_TX_OFFERED_MBPS_MIN)
            | (packets >= 100)
        )
    )

    low_rate = df["source"].eq(SIMULATOR_SOURCE) & ~load_bearing

    traffic_class = df["traffic_class"].fillna("").astype(str).copy()
    traffic_class.loc[load_bearing] = LOAD_BEARING_TRAFFIC_CLASS
    traffic_class.loc[low_rate] = LOW_RATE_TRAFFIC_CLASS

    return traffic_class


def simulator_comparison_frame(combined_df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_simulator_schema(combined_df)

    simulator_rows = out["source"].eq(SIMULATOR_SOURCE)
    load_bearing_rows = out["traffic_class"].eq(LOAD_BEARING_TRAFFIC_CLASS)

    return out.loc[(~simulator_rows) | load_bearing_rows].copy()


def assign_offered_load_label(value: object) -> str:
    if pd.isna(value):
        return "unassigned"

    value_text = str(value).strip().lower()

    if value_text in {"low", "light"}:
        return "low"

    if value_text in {"medium", "moderate"}:
        return "medium"

    if value_text in {"high", "heavy"}:
        return "high"

    value_number = parse_number(value_text)

    if pd.isna(value_number):
        return "unassigned"

    if value_number > 1.0:
        value_number = value_number / 100.0

    if value_number <= 0.33:
        return "low"

    if value_number <= 0.66:
        return "medium"

    return "high"


def assign_offered_load_labels_from_series(series: pd.Series) -> pd.Series:
    values = clean_numeric(series)
    labels = pd.Series("unassigned", index=series.index, dtype="object")

    valid = values.dropna()

    if valid.empty:
        return labels

    low_cutoff = valid.quantile(1 / 3)
    high_cutoff = valid.quantile(2 / 3)

    labels.loc[values <= low_cutoff] = "low"
    labels.loc[(values > low_cutoff) & (values <= high_cutoff)] = "medium"
    labels.loc[values > high_cutoff] = "high"

    return labels


def build_synnetqos_simulator_subset(df: pd.DataFrame) -> pd.DataFrame:
    throughput_col = first_existing_column(
        df,
        [
            "Download_Speed_Mbps",
            "Download_Throughput_Mbps",
            "Throughput_Mbps",
            "throughput_mbps",
        ],
        "SynNetQoS simulator subset",
    )
    delay_col = first_existing_column(
        df,
        ["Latency_ms", "Delay_ms", "mean_delay_ms"],
        "SynNetQoS simulator subset",
    )
    jitter_col = first_existing_column(
        df,
        ["Jitter_ms", "mean_jitter_ms"],
        "SynNetQoS simulator subset",
    )
    offered_downlink_col = first_existing_column(
        df,
        ["Offered_Downlink_Mbps", "offered_downlink_mbps", "tx_offered_mbps"],
        "SynNetQoS simulator subset",
        required=False,
    )
    network_col = first_existing_column(
        df,
        ["Network_Type", "network_type"],
        "SynNetQoS simulator subset",
        required=False,
    )
    fallback_load_col = first_existing_column(
        df,
        ["Congestion_Level", "congestion_level", "Tower_Load", "tower_load"],
        "SynNetQoS simulator subset",
        required=False,
    )
    infrastructure_col = first_existing_column(
        df,
        ["Infrastructure_Profile", "infrastructure_profile"],
        "SynNetQoS simulator subset",
        required=False,
    )
    movement_col = first_existing_column(
        df,
        ["Movement_Speed", "movement_speed"],
        "SynNetQoS simulator subset",
        required=False,
    )
    signal_col = first_existing_column(
        df,
        ["Signal_Strength_dBm", "signal_strength_dbm", "rsrp_dbm"],
        "SynNetQoS simulator subset",
        required=False,
    )
    packet_loss_col = first_existing_column(
        df,
        ["Packet_Loss_Ratio", "Packet_Loss_Rate", "packet_loss_ratio", "packet_loss_rate"],
        "SynNetQoS simulator subset",
        required=False,
    )

    subset = df.copy()

    if network_col is not None:
        subset = subset[subset[network_col].isin(["5G NSA", "5G SA"])].copy()

    if infrastructure_col is not None:
        subset = subset[subset[infrastructure_col] == "Nominal"].copy()

    if movement_col is not None:
        subset = subset[subset[movement_col] == "Static"].copy()

    if fallback_load_col is not None:
        subset = subset[subset[fallback_load_col].isin(["Low", "Medium"])].copy()

    if signal_col is not None:
        subset = subset[clean_numeric(subset[signal_col]) > -110].copy()

    if offered_downlink_col is not None:
        load_labels = assign_offered_load_labels_from_series(subset[offered_downlink_col])
    else:
        load_labels = (
            subset[fallback_load_col].map(assign_offered_load_label)
            if fallback_load_col is not None
            else pd.Series("unassigned", index=subset.index, dtype="object")
        )

    out = pd.DataFrame(
        {
            "source": SYNNETQOS_SOURCE,
            "scenario_id": "synnetqos_5g_simulator_comparable_subset",
            "raw_file": "",
            "load_label": load_labels.to_numpy(),
            "seed": np.nan,
            "flow_id": np.nan,
            "protocol": "",
            "endpoint": "",
            "traffic_class": SYNNETQOS_TRAFFIC_CLASS,
            "tx_packets": np.nan,
            "rx_packets": np.nan,
            "tx_bytes": np.nan,
            "rx_bytes": np.nan,
            "tx_offered_mbps": (
                clean_numeric(subset[offered_downlink_col]).to_numpy()
                if offered_downlink_col is not None
                else np.nan
            ),
            "throughput_mbps": clean_numeric(subset[throughput_col]),
            "mean_delay_ms": clean_numeric(subset[delay_col]),
            "mean_jitter_ms": clean_numeric(subset[jitter_col]),
            "mean_flow_throughput_mbps": np.nan,
            "mean_flow_delay_ms": np.nan,
        }
    )

    if packet_loss_col is not None:
        packet_loss = clean_numeric(subset[packet_loss_col])
        out["packet_loss_ratio"] = packet_loss.where(packet_loss <= 1.0, packet_loss / 100.0)
    else:
        out["packet_loss_ratio"] = np.nan

    return ensure_simulator_schema(out)


def first_existing_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    dataset_name: str,
    required: bool = True,
) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise KeyError(f"{dataset_name} is missing all expected columns from: {list(candidates)}")

    return None


def summary_stats(df: pd.DataFrame, group_cols: Sequence[str], value_cols: Sequence[str]) -> pd.DataFrame:
    require_columns(df, [*group_cols, *value_cols], "simulator comparison input")

    rows: list[dict[str, object]] = []

    for group_key, group in df.groupby(list(group_cols), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        base = dict(zip(group_cols, group_key, strict=False))

        for col in value_cols:
            values = clean_numeric(group[col]).dropna()

            rows.append(
                {
                    **base,
                    "kpi": col,
                    "n": int(values.shape[0]),
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "median": float(values.median()) if not values.empty else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "q1": float(values.quantile(0.25)) if not values.empty else np.nan,
                    "q3": float(values.quantile(0.75)) if not values.empty else np.nan,
                    "min": float(values.min()) if not values.empty else np.nan,
                    "max": float(values.max()) if not values.empty else np.nan,
                }
            )

    return pd.DataFrame(rows)


def simulator_kpi_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    return summary_stats(
        simulator_comparison_frame(combined_df),
        ["source", "load_label"],
        KPI_COLUMNS,
    )


def compare_simulator_kpis(combined_df: pd.DataFrame) -> pd.DataFrame:
    summary = simulator_kpi_summary(combined_df)
    rows: list[dict[str, object]] = []

    load_labels = sorted(summary["load_label"].dropna().astype(str).unique())

    for load_label in load_labels:
        for kpi in CROSS_SOURCE_KPI_COLUMNS:
            syn_row = matching_summary_row(summary, SYNNETQOS_SOURCE, load_label, kpi)
            sim_row = matching_summary_row(summary, SIMULATOR_SOURCE, load_label, kpi)

            syn_median = syn_row.get("median", np.nan)
            sim_median = sim_row.get("median", np.nan)

            rows.append(
                {
                    "load_label": load_label,
                    "kpi": kpi,
                    "synnetqos_n": int(syn_row.get("n", 0)),
                    "simulator_n": int(sim_row.get("n", 0)),
                    "synnetqos_median": syn_median,
                    "simulator_median": sim_median,
                    "median_absolute_delta_synnetqos_minus_simulator": safe_delta(syn_median, sim_median),
                    "median_ratio_synnetqos_over_simulator": safe_ratio(syn_median, sim_median),
                    "median_percent_delta_vs_simulator": safe_percent_delta(syn_median, sim_median),
                    "comparison_scope": "KPI-level simulator-trace comparison by offered-load label",
                }
            )

    return pd.DataFrame(rows)


def simulator_comparison_interpretation_flags(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in comparison_df.iterrows():
        if row["simulator_n"] == 0 or row["synnetqos_n"] == 0:
            rows.append(
                {
                    "load_label": row["load_label"],
                    "kpi": row["kpi"],
                    "severity": "review",
                    "flag": "missing_comparison_side",
                    "interpretation": "One side has no comparable rows for this load label and KPI.",
                }
            )
            continue

        if row["simulator_n"] < 3:
            rows.append(
                {
                    "load_label": row["load_label"],
                    "kpi": row["kpi"],
                    "severity": "note",
                    "flag": "small_simulator_sample",
                    "interpretation": "The simulator side has fewer than three parsed KPI rows.",
                }
            )

        ratio = row["median_ratio_synnetqos_over_simulator"]

        if pd.isna(ratio):
            rows.append(
                {
                    "load_label": row["load_label"],
                    "kpi": row["kpi"],
                    "severity": "note",
                    "flag": "ratio_not_available",
                    "interpretation": "The median ratio could not be computed.",
                }
            )
        elif row["kpi"] == "throughput_mbps" and not 0.25 <= ratio <= 4.0:
            rows.append(
                {
                    "load_label": row["load_label"],
                    "kpi": row["kpi"],
                    "severity": "review",
                    "flag": "large_throughput_scale_difference",
                    "interpretation": "Throughput medians differ by more than the first-pass review band.",
                }
            )
        elif row["kpi"] in {"mean_delay_ms", "mean_jitter_ms"} and not 0.10 <= ratio <= 10.0:
            rows.append(
                {
                    "load_label": row["load_label"],
                    "kpi": row["kpi"],
                    "severity": "review",
                    "flag": "large_delay_or_jitter_scale_difference",
                    "interpretation": "Delay or jitter medians differ by more than the first-pass review band.",
                }
            )

    if not rows:
        return pd.DataFrame(
            [
                {
                    "load_label": "all",
                    "kpi": "all",
                    "severity": "note",
                    "flag": "no_first_pass_flags",
                    "interpretation": "No first-pass review flags were triggered. This is not a validation claim.",
                }
            ]
        )

    return pd.DataFrame(rows)



def simulator_packet_loss_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    summary = simulator_kpi_summary(combined_df)
    packet_loss = summary[
        (summary["source"] == SIMULATOR_SOURCE)
        & (summary["kpi"] == "packet_loss_ratio")
    ].copy()

    if packet_loss.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "load_label",
                "kpi",
                "n",
                "mean",
                "median",
                "std",
                "q1",
                "q3",
                "min",
                "max",
                "comparison_scope",
            ]
        )

    packet_loss["comparison_scope"] = (
        "Simulator-side packet-loss diagnostic only; SynNetQoS does not expose a "
        "packet-level Tx/Rx packet-loss counterpart in this comparison branch."
    )
    return packet_loss.reset_index(drop=True)


def simulator_kpi_trend_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    summary = simulator_kpi_summary(combined_df)
    rows: list[dict[str, object]] = []

    for kpi in CROSS_SOURCE_KPI_COLUMNS:
        for source in [SIMULATOR_SOURCE, SYNNETQOS_SOURCE]:
            source_summary = summary[
                (summary["source"] == source)
                & (summary["kpi"] == kpi)
            ].copy()
            source_summary["load_order"] = source_summary["load_label"].astype(str).map(LOAD_LABEL_ORDER)
            source_summary = source_summary.dropna(subset=["load_order", "median"]).sort_values("load_order")

            medians = source_summary["median"].astype(float).to_numpy()
            if len(medians) < 2:
                direction = "insufficient_data"
                low_to_high_ratio = np.nan
            else:
                diffs = np.diff(medians)
                if np.all(diffs > 0):
                    direction = "increasing"
                elif np.all(diffs < 0):
                    direction = "decreasing"
                elif np.all(np.isclose(diffs, 0.0)):
                    direction = "flat"
                else:
                    direction = "mixed"

                low_to_high_ratio = safe_ratio(medians[-1], medians[0])

            rows.append(
                {
                    "source": source,
                    "kpi": kpi,
                    "n_load_labels": int(source_summary["load_label"].nunique()),
                    "load_labels": ";".join(source_summary["load_label"].astype(str).tolist()),
                    "median_low": median_for_label(source_summary, "low"),
                    "median_medium": median_for_label(source_summary, "medium"),
                    "median_high": median_for_label(source_summary, "high"),
                    "low_to_high_ratio": low_to_high_ratio,
                    "trend_direction": direction,
                    "scope_note": "Load-response diagnostic only; this is not a calibration or equivalence score.",
                }
            )

    out = pd.DataFrame(rows)
    comparison_rows: list[dict[str, object]] = []

    for kpi in CROSS_SOURCE_KPI_COLUMNS:
        syn = out[(out["source"] == SYNNETQOS_SOURCE) & (out["kpi"] == kpi)]
        sim = out[(out["source"] == SIMULATOR_SOURCE) & (out["kpi"] == kpi)]

        if syn.empty or sim.empty:
            continue

        syn_direction = str(syn.iloc[0]["trend_direction"])
        sim_direction = str(sim.iloc[0]["trend_direction"])
        monotonic_directions = {"increasing", "decreasing"}

        if syn_direction == sim_direction and syn_direction in monotonic_directions:
            alignment = "matched_monotonic"
            note_prefix = "Both sources show the same monotonic load response"
        elif syn_direction in monotonic_directions and sim_direction in monotonic_directions:
            alignment = "not_matched"
            note_prefix = "The two sources show different monotonic load responses"
        else:
            alignment = "ambiguous"
            note_prefix = "At least one source has a mixed, flat, or insufficient load response"

        comparison_rows.append(
            {
                "source": "trend_comparison",
                "kpi": kpi,
                "n_load_labels": min(int(syn.iloc[0]["n_load_labels"]), int(sim.iloc[0]["n_load_labels"])),
                "load_labels": "low;medium;high",
                "median_low": np.nan,
                "median_medium": np.nan,
                "median_high": np.nan,
                "low_to_high_ratio": np.nan,
                "trend_direction": alignment,
                "scope_note": (
                    f"{note_prefix}: SynNetQoS trend={syn_direction}; "
                    f"simulator trend={sim_direction}. Trend comparison is diagnostic and "
                    "does not imply absolute-value equivalence."
                ),
            }
        )

    if comparison_rows:
        out = pd.concat([out, pd.DataFrame(comparison_rows)], ignore_index=True)

    return out


def simulator_comparison_verdict(
    comparison_df: pd.DataFrame,
    trend_summary_df: pd.DataFrame,
    interpretation_flags_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    throughput_trend = comparison_trend_status(trend_summary_df, "throughput_mbps")
    delay_trend = comparison_trend_status(trend_summary_df, "mean_delay_ms")
    jitter_trend = comparison_trend_status(trend_summary_df, "mean_jitter_ms")

    throughput_flags = flags_for_kpi(interpretation_flags_df, "throughput_mbps")
    delay_flags = flags_for_kpi(interpretation_flags_df, "mean_delay_ms")
    jitter_flags = flags_for_kpi(interpretation_flags_df, "mean_jitter_ms")

    rows.append(
        {
            "component": "throughput",
            "status": "usable_diagnostic",
            "recommended_placement": "main_or_supplementary",
            "summary": (
                "Throughput is the strongest KPI-level simulator-reference comparison. "
                f"Load-response trend status: {throughput_trend}."
            ),
            "claim_boundary": (
                "Use as a throughput-scale/load-response sanity check only; do not claim "
                "5G-LENA/ns-3 validation or packet-level equivalence."
            ),
            "supporting_flags": throughput_flags,
        }
    )

    rows.append(
        {
            "component": "delay",
            "status": "diagnostic_only",
            "recommended_placement": "supplementary_or_diagnostic",
            "summary": (
                "Delay remains a diagnostic output because the simulator and SynNetQoS "
                f"operate at different abstraction levels. Load-response trend status: {delay_trend}."
            ),
            "claim_boundary": (
                "Do not treat delay as aligned. High-load ns-3 queueing can dominate the "
                "absolute scale, while SynNetQoS reports session-level KPI latency."
            ),
            "supporting_flags": delay_flags,
        }
    )

    rows.append(
        {
            "component": "jitter",
            "status": "diagnostic_only",
            "recommended_placement": "supplementary_or_diagnostic",
            "summary": (
                "Jitter remains a diagnostic output because absolute scale and measurement "
                f"semantics differ across the two systems. Load-response trend status: {jitter_trend}."
            ),
            "claim_boundary": "Do not use jitter as evidence of simulator equivalence or calibration.",
            "supporting_flags": jitter_flags,
        }
    )

    rows.append(
        {
            "component": "packet_loss",
            "status": "simulator_side_only",
            "recommended_placement": "supplementary_or_diagnostic",
            "summary": (
                "Packet loss is retained as a simulator-side diagnostic because SynNetQoS "
                "does not expose a packet-level Tx/Rx packet-loss counterpart in this branch."
            ),
            "claim_boundary": "Do not make cross-source packet-loss comparison claims.",
            "supporting_flags": "not_applicable",
        }
    )

    rows.append(
        {
            "component": "overall",
            "status": "supplementary_simulator_reference",
            "recommended_placement": "supplementary_with_cautious_main_text",
            "summary": (
                "The 5G-LENA/ns-3 branch is suitable as a controlled KPI-level simulator-reference "
                "diagnostic. It strengthens auditability but is not validation, calibration, or "
                "full PHY/MAC equivalence."
            ),
            "claim_boundary": (
                "Phrase as a controlled KPI-level simulator-trace comparison. Avoid claims that "
                "SynNetQoS reproduces ns-3 behavior."
            ),
            "supporting_flags": summarize_all_flags(interpretation_flags_df),
        }
    )

    return pd.DataFrame(rows)


def comparison_trend_status(trend_summary_df: pd.DataFrame, kpi: str) -> str:
    matched = trend_summary_df[
        (trend_summary_df["source"] == "trend_comparison")
        & (trend_summary_df["kpi"] == kpi)
    ]

    if matched.empty:
        return "not_available"

    return str(matched.iloc[0]["trend_direction"])


def flags_for_kpi(flags_df: pd.DataFrame, kpi: str) -> str:
    if flags_df.empty or "kpi" not in flags_df.columns:
        return "none"

    matched = flags_df[flags_df["kpi"] == kpi]
    if matched.empty:
        return "none"

    return "; ".join(
        f"{row.severity}:{row.flag}"
        for row in matched.itertuples(index=False)
    )


def summarize_all_flags(flags_df: pd.DataFrame) -> str:
    if flags_df.empty or "flag" not in flags_df.columns:
        return "none"

    return "; ".join(
        f"{row.kpi}:{row.severity}:{row.flag}"
        for row in flags_df.itertuples(index=False)
    )


def median_for_label(summary: pd.DataFrame, load_label: str) -> float:
    matched = summary[summary["load_label"].astype(str) == load_label]
    if matched.empty:
        return np.nan

    return float(matched.iloc[0]["median"])

def simulator_feature_mapping_table() -> pd.DataFrame:
    rows = [
        {
            "synnetqos_feature": "Download_Speed_Mbps",
            "simulator_feature": "FlowMonitor throughput",
            "comparison_kpi": "throughput_mbps",
            "unit": "Mbps",
            "mapping_type": "KPI-level",
            "notes": "Summary comparison only; not packet-level equivalence.",
        },
        {
            "synnetqos_feature": "Latency_ms",
            "simulator_feature": "FlowMonitor mean delay",
            "comparison_kpi": "mean_delay_ms",
            "unit": "ms",
            "mapping_type": "KPI-level",
            "notes": "Compares mean delay summaries, not full delay traces.",
        },
        {
            "synnetqos_feature": "Jitter_ms",
            "simulator_feature": "FlowMonitor mean jitter",
            "comparison_kpi": "mean_jitter_ms",
            "unit": "ms",
            "mapping_type": "KPI-level",
            "notes": "Compares mean jitter summaries only.",
        },
        {
            "synnetqos_feature": "Packet loss field if present",
            "simulator_feature": "Derived from Tx/Rx packet counts",
            "comparison_kpi": "packet_loss_ratio",
            "unit": "ratio",
            "mapping_type": "Derived KPI-level",
            "notes": "Simulator-side diagnostic only unless a comparable SynNetQoS packet-loss field is explicitly present.",
        },
    ]

    return pd.DataFrame(rows)


def plot_simulator_throughput_comparison(combined_df: pd.DataFrame) -> plt.Figure:
    return plot_simulator_kpi_bar(
        combined_df,
        kpi="throughput_mbps",
        ylabel="Throughput (Mbps)",
    )


def plot_simulator_delay_jitter_comparison(combined_df: pd.DataFrame) -> plt.Figure:
    plot_df = build_plot_summary(combined_df, ["mean_delay_ms", "mean_jitter_ms"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    plot_specs = [
        ("mean_delay_ms", "Median mean delay (ms)", axes[0]),
        ("mean_jitter_ms", "Median mean jitter (ms)", axes[1]),
    ]

    for kpi, ylabel, ax in plot_specs:
        subset = plot_df[plot_df["kpi"] == kpi].copy()

        if subset.empty:
            ax.text(0.5, 0.5, f"No {kpi} data available", transform=ax.transAxes, ha="center", va="center")
        else:
            sns.barplot(
                data=subset,
                x="load_label",
                y="median",
                hue="source",
                ax=ax,
            )

        ax.set_xlabel("Offered-load label")
        ax.set_ylabel(ylabel)
        ax.legend(title="Source")

    return fig


def plot_simulator_kpi_bar(combined_df: pd.DataFrame, kpi: str, ylabel: str) -> plt.Figure:
    plot_df = build_plot_summary(combined_df, [kpi])
    fig, ax = plt.subplots(figsize=(8, 5))

    if plot_df.empty:
        ax.text(0.5, 0.5, f"No {kpi} comparison data available", transform=ax.transAxes, ha="center", va="center")
    else:
        sns.barplot(
            data=plot_df,
            x="load_label",
            y="median",
            hue="source",
            ax=ax,
        )

    ax.set_xlabel("Offered-load label")
    ax.set_ylabel(ylabel)
    ax.legend(title="Source")
    fig.tight_layout()
    return fig


def build_plot_summary(combined_df: pd.DataFrame, kpis: Sequence[str]) -> pd.DataFrame:
    summary = summary_stats(
        simulator_comparison_frame(combined_df),
        ["source", "load_label"],
        kpis,
    )
    order = {"low": 0, "medium": 1, "high": 2, "unassigned": 3}
    summary["load_order"] = summary["load_label"].map(order).fillna(99)
    return summary.sort_values(["load_order", "source", "kpi"]).drop(columns=["load_order"])


def matching_summary_row(summary: pd.DataFrame, source: str, load_label: str, kpi: str) -> dict[str, object]:
    matched = summary[
        (summary["source"] == source)
        & (summary["load_label"].astype(str) == str(load_label))
        & (summary["kpi"] == kpi)
    ]

    if matched.empty:
        return {"n": 0, "median": np.nan}

    return matched.iloc[0].to_dict()


def safe_delta(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right):
        return np.nan

    return float(left - right)


def safe_ratio(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right) or right == 0:
        return np.nan

    return float(left / right)


def safe_percent_delta(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right) or right == 0:
        return np.nan

    return float(100.0 * (left - right) / abs(right))
