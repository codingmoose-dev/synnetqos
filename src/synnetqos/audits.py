import pandas as pd

def dataset_integrity_summary(df: pd.DataFrame, expected_sessions: int, session_length: int) -> pd.DataFrame:
    expected_rows = expected_sessions * session_length
    timestamp_check = df.sort_values(["Session_ID", "Timestamp"]).groupby("Session_ID")["Timestamp"].apply(lambda x: x.is_monotonic_increasing)
    
    # Packages core integrity stats into a DataFrame
    summary_data = {
        "Metric": ["Expected_Rows", "Generated_Rows", "Monotonic_Timestamps_Percent"],
        "Value": [expected_rows, len(df), round(timestamp_check.mean() * 100, 2)]
    }
    return pd.DataFrame(summary_data)

def numerical_range_summary(df: pd.DataFrame) -> pd.DataFrame:
    range_cols = [
        "Signal_Strength_dBm", "Link_Capacity_Downlink_Mbps", "Offered_Downlink_Mbps",
        "Download_Speed_Mbps", "Link_Capacity_Upload_Mbps", "Offered_Upload_Mbps",
        "Upload_Speed_Mbps", "Latency_ms", "Jitter_ms", "Battery_Level_percent",
        "Interval_Handover_Count", "Cumulative_Handover_Count", "Distance_to_Tower_km"
    ]
    
    valid_cols = [col for col in range_cols if col in df.columns]
    # Computes min, mean, max and returns a transposed dataframe for readable CSV structure
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
    