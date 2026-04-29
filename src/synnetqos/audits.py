from __future__ import annotations

import pandas as pd


RANGE_COLUMNS = [
    "Signal_Strength_dBm",
    "Link_Capacity_Downlink_Mbps",
    "Offered_Downlink_Mbps",
    "Download_Speed_Mbps",
    "Link_Capacity_Upload_Mbps",
    "Offered_Upload_Mbps",
    "Upload_Speed_Mbps",
    "Latency_ms",
    "Jitter_ms",
    "Battery_Level_percent",
    "Interval_Handover_Count",
    "Cumulative_Handover_Count",
    "Distance_to_Tower_km",
]


def dataset_integrity_summary(df: pd.DataFrame, num_sessions: int, session_length: int) -> pd.DataFrame:
    expected_rows = num_sessions * session_length
    session_lengths = df.groupby("Session_ID").size()

    monotonic = (
        df.sort_values(["Session_ID", "Timestamp"])
          .groupby("Session_ID")["Timestamp"]
          .apply(lambda s: s.is_monotonic_increasing)
    )

    rows = [
        {"check": "expected_rows", "value": expected_rows},
        {"check": "generated_rows", "value": len(df)},
        {"check": "unique_sessions", "value": df["Session_ID"].nunique()},
        {"check": "min_session_length", "value": int(session_lengths.min())},
        {"check": "max_session_length", "value": int(session_lengths.max())},
        {"check": "monotonic_timestamp_fraction", "value": float(monotonic.mean())},
    ]
    return pd.DataFrame(rows)


def numerical_range_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in RANGE_COLUMNS if c in df.columns]
    return df[cols].agg(["min", "mean", "max"]).round(4).T.reset_index(names="variable")


def dataset_schema(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_fraction": [float(df[c].isna().mean()) for c in df.columns],
    })
