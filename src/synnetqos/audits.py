from __future__ import annotations
import pandas as pd

def dataset_integrity_summary(df: pd.DataFrame, expected_sessions: int, session_length: int) -> pd.DataFrame:
    expected_rows = expected_sessions * session_length
    timestamp_check = df.sort_values(["Session_ID", "Timestamp"]).groupby("Session_ID")["Timestamp"].apply(lambda x: x.is_monotonic_increasing)

    summary_data = {
        "Metric": ["Expected_Rows", "Generated_Rows", "Monotonic_Timestamps_Percent"],
        "Value": [expected_rows, len(df), round(timestamp_check.mean() * 100, 2)]
    }
    return pd.DataFrame(summary_data)

def numerical_range_summary(df: pd.DataFrame) -> pd.DataFrame:
    range_cols = [
        "Signal_Strength_dBm", "Signal_Strength_Unclipped_dBm", "Carrier_Frequency_GHz",
        "Distance_2D_m", "Distance_3D_m", "Breakpoint_Distance_m", "LOS_Probability",
        "Path_Loss_dB", "Deterministic_Path_Loss_dB", "Shadow_Fading_dB", "Fast_Fading_dB",
        "Obstruction_Penalty_dB", "Weather_Penalty_dB", "Mobility_Penalty_dB", "Indoor_Penalty_dB",
        "Contextual_Penalty_dB", "Effective_TX_Power_dBm", "Link_Capacity_Downlink_Mbps",
        "Offered_Downlink_Mbps", "Download_Speed_Mbps", "Link_Capacity_Upload_Mbps",
        "Offered_Upload_Mbps", "Upload_Speed_Mbps", "Latency_ms", "Jitter_ms",
        "Battery_Level_percent", "Interval_Handover_Count", "Cumulative_Handover_Count",
        "Distance_to_Tower_km"
    ]
    
    valid_cols = [col for col in range_cols if col in df.columns]
    summary = df[valid_cols].agg(["min", "mean", "max"]).round(2).T.reset_index()
    summary.rename(columns={"index": "Feature"}, inplace=True)
    return summary

def outlier_report(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    outlier_stats = []
    
    for col in features:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        
        outlier_stats.append({
            "Feature": col, 
            "Lower_Bound": round(lower_bound, 2),
            "Upper_Bound": round(upper_bound, 2),
            "Outlier_Count": count, 
            "Outlier_Percent": round((count/len(df))*100, 2)
        })
        
    return pd.DataFrame(outlier_stats)

def dataset_schema(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(df[col].dtype) for col in df.columns],
        "Missing_Count": [int(df[col].isna().sum()) for col in df.columns],
        "Missing_Percent": [round(float(df[col].isna().mean() * 100), 4) for col in df.columns],
    })

def propagation_model_audit(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Propagation_Model", "Propagation_Scenario", "LOS_State", "Band", "Carrier_Frequency_GHz",
        "Distance_2D_m", "Path_Loss_dB", "Signal_Strength_dBm", "RSRP_Clipped"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Propagation audit is missing required columns: {missing}")

    rows = []
    group_cols = ["Propagation_Model", "Propagation_Scenario", "LOS_State"]
    for keys, group in df.groupby(group_cols, dropna=False):
        rows.append({
            "Propagation_Model": keys[0],
            "Propagation_Scenario": keys[1],
            "LOS_State": keys[2],
            "Rows": int(len(group)),
            "Median_Carrier_Frequency_GHz": round(pd.to_numeric(group["Carrier_Frequency_GHz"], errors="coerce").median(), 3),
            "Median_Distance_2D_m": round(pd.to_numeric(group["Distance_2D_m"], errors="coerce").median(), 2),
            "Median_Path_Loss_dB": round(pd.to_numeric(group["Path_Loss_dB"], errors="coerce").median(), 2),
            "Median_RSRP_dBm": round(pd.to_numeric(group["Signal_Strength_dBm"], errors="coerce").median(), 2),
            "RSRP_Clipped_Percent": round(group["RSRP_Clipped"].astype(bool).mean() * 100, 2),
        })

    return pd.DataFrame(rows)

def _categorical_drop_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if column not in df.columns:
        return pd.DataFrame(rows)

    for value, group in df.groupby(column, dropna=False):
        positives = int(group["Dropped_Connection"].astype(bool).sum())
        total = int(len(group))
        rows.append(
            {
                "summary_type": f"by_{column}",
                "category": str(value),
                "rows": total,
                "positive_count": positives,
                "positive_rate": round(float(positives / total), 6) if total else 0.0,
            }
        )

    return pd.DataFrame(rows)

def drop_event_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Session_ID", "Connected_Duration_min", "Dropped_Connection"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Drop-event audit is missing required columns: {missing}")

    ordered = df.sort_values(["Session_ID", "Connected_Duration_min"]).copy()
    ordered["Drop_t_plus_1"] = ordered.groupby("Session_ID")["Dropped_Connection"].shift(-1)

    dropped = ordered["Dropped_Connection"].astype(bool)
    future = ordered["Drop_t_plus_1"].dropna().astype(bool)

    rows: list[dict[str, object]] = [
        {
            "summary_type": "overall_current_drop",
            "category": "all_rows",
            "rows": int(len(dropped)),
            "positive_count": int(dropped.sum()),
            "positive_rate": round(float(dropped.mean()), 6),
        },
        {
            "summary_type": "overall_future_drop",
            "category": "nonterminal_session_rows",
            "rows": int(len(future)),
            "positive_count": int(future.sum()),
            "positive_rate": round(float(future.mean()), 6),
        },
    ]

    summary_frames = [pd.DataFrame(rows)]
    for column in [
        "Network_Type",
        "Congestion_Level",
        "Tower_Load",
        "Infrastructure_Profile",
        "Band",
        "Movement_Speed",
        "Obstruction_Level",
        "Is_Indoor",
    ]:
        frame = _categorical_drop_summary(ordered, column)
        if not frame.empty:
            summary_frames.append(frame)

    return pd.concat(summary_frames, ignore_index=True)
