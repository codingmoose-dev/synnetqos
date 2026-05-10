from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, spearmanr, ttest_ind, wasserstein_distance

COMMON_EXTERNAL_COLUMNS: tuple[str, ...] = ("source", "network_type", "rsrp_dbm", "download_mbps", "upload_mbps", "latency_ms", "jitter_ms")
PAPER_FACING_EXTERNAL_VARIABLES: tuple[str, ...] = ("rsrp_dbm", "download_mbps", "jitter_ms")
SUPPLEMENTARY_EXTERNAL_VARIABLES: tuple[str, ...] = ("upload_mbps", "latency_ms")
SUMMARY_COLUMNS: tuple[str, ...] = ("dataset", "variable", "n", "mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max", "iqr")
COMPARISON_COLUMNS: tuple[str, ...] = ("comparison_group", "variable", "synthetic_network_type", "external_network_type", "synthetic_column", "external_column", "synthetic_n", "external_n", "synthetic_median", "external_median", "median_difference", "median_ratio", "synthetic_iqr", "external_iqr", "iqr_ratio", "ks_stat", "wasserstein_distance", "js_distance")

# Raise a clear error if required columns are missing.
def require_columns(df: pd.DataFrame, required: Sequence[str], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing: raise KeyError(f"{dataset_name} is missing required columns: {missing}")

# Return the first matching column name from a list of aliases.
def first_existing_column(df: pd.DataFrame, candidates: Sequence[str], dataset_name: str, logical_name: str) -> str:
    for col in candidates:
        if col in df.columns: return col
    raise KeyError(f"{dataset_name} is missing a column for {logical_name}. Tried aliases: {list(candidates)}")

# Convert a series to finite numeric values only.
def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

# Ensure that an aligned external-comparison table uses the common schema.
def ensure_common_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in COMMON_EXTERNAL_COLUMNS:
        if col not in out.columns: out[col] = np.nan
    return out.loc[:, list(COMMON_EXTERNAL_COLUMNS)]

# Map SynNetQoS output columns into the common external-comparison schema.
def align_synnetqos(syn_df: pd.DataFrame) -> pd.DataFrame:
    required = ["Network_Type", "Signal_Strength_dBm", "Download_Speed_Mbps", "Upload_Speed_Mbps", "Latency_ms", "Jitter_ms"]
    require_columns(syn_df, required, "SynNetQoS")
    out = pd.DataFrame({"source": "SynNetQoS", "network_type": syn_df["Network_Type"].astype(str), "rsrp_dbm": pd.to_numeric(syn_df["Signal_Strength_dBm"], errors="coerce"), "download_mbps": pd.to_numeric(syn_df["Download_Speed_Mbps"], errors="coerce"), "upload_mbps": pd.to_numeric(syn_df["Upload_Speed_Mbps"], errors="coerce"), "latency_ms": pd.to_numeric(syn_df["Latency_ms"], errors="coerce"), "jitter_ms": pd.to_numeric(syn_df["Jitter_ms"], errors="coerce")})
    return ensure_common_schema(out)

# Return an audit table for the controlled 5G subset filters.
def controlled_5g_subset_audit(syn_df: pd.DataFrame) -> pd.DataFrame:
    required = ["Network_Type", "Infrastructure_Profile", "Movement_Speed", "Congestion_Level", "Signal_Strength_dBm"]
    require_columns(syn_df, required, "SynNetQoS controlled 5G subset")
    current, rows = syn_df.copy(), [{"step": "input_rows", "filter_applied": "none", "rows_remaining": int(len(syn_df))}]
    filters = [("5g_network_type", "Network_Type in ['5G NSA', '5G SA']", current["Network_Type"].isin(["5G NSA", "5G SA"])), ("nominal_infrastructure", "Infrastructure_Profile == 'Nominal'", current["Infrastructure_Profile"] == "Nominal"), ("static_mobility", "Movement_Speed == 'Static'", current["Movement_Speed"] == "Static"), ("low_or_medium_congestion", "Congestion_Level in ['Low', 'Medium']", current["Congestion_Level"].isin(["Low", "Medium"])), ("usable_signal", "Signal_Strength_dBm > -110", pd.to_numeric(current["Signal_Strength_dBm"], errors="coerce") > -110)]
    cumulative_mask = pd.Series(True, index=current.index)
    for step, description, mask in filters:
        cumulative_mask = cumulative_mask & mask.fillna(False)
        rows.append({"step": step, "filter_applied": description, "rows_remaining": int(cumulative_mask.sum())})
    return pd.DataFrame(rows)

# Return the controlled 5G-like SynNetQoS subset in the common schema.
def controlled_5g_synnetqos_subset(syn_df: pd.DataFrame) -> pd.DataFrame:
    required = ["Network_Type", "Infrastructure_Profile", "Movement_Speed", "Congestion_Level", "Signal_Strength_dBm"]
    require_columns(syn_df, required, "SynNetQoS controlled 5G subset")
    mask = (syn_df["Network_Type"].isin(["5G NSA", "5G SA"]) & (syn_df["Infrastructure_Profile"] == "Nominal") & (syn_df["Movement_Speed"] == "Static") & (syn_df["Congestion_Level"].isin(["Low", "Medium"])) & (pd.to_numeric(syn_df["Signal_Strength_dBm"], errors="coerce") > -110))
    out = align_synnetqos(syn_df.loc[mask].copy())
    out["source"], out["network_type"] = "SynNetQoS_controlled_5G_subset", "5G"
    return ensure_common_schema(out)

# Return the controlled 5G subset and the corresponding filter audit.
def build_controlled_5g_synnetqos(syn_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return controlled_5g_synnetqos_subset(syn_df), controlled_5g_subset_audit(syn_df)

# Map Vienna phone LTE/5G measurements into the common schema.
def align_vienna_phone(phone_lte: pd.DataFrame, phone_5g: pd.DataFrame) -> pd.DataFrame:
    lte_rsrp_col = first_existing_column(phone_lte, ["rsrp_dbm", "rsrp", "RSRP"], "Vienna phone LTE", "RSRP")
    lte_dl_col = first_existing_column(phone_lte, ["dl_throughput_mbps", "download_mbps", "dl_mbps"], "Vienna phone LTE", "download throughput")
    lte_ul_col = first_existing_column(phone_lte, ["ul_throughput_mbps", "upload_mbps", "ul_mbps"], "Vienna phone LTE", "upload throughput")
    fiveg_rsrp_col = first_existing_column(phone_5g, ["rsrp_dbm", "rsrp", "RSRP"], "Vienna phone 5G", "RSRP")
    fiveg_dl_col = first_existing_column(phone_5g, ["dl_throughput_mbps", "download_mbps", "dl_mbps"], "Vienna phone 5G", "download throughput")
    fiveg_ul_col = first_existing_column(phone_5g, ["ul_throughput_mbps", "upload_mbps", "ul_mbps"], "Vienna phone 5G", "upload throughput")
    phone_5g_used, fiveg_source = (
        phone_5g[phone_5g["beam_type"].astype(str).str.contains("Serving", case=False, na=False)].copy(),
        "Vienna_phone_5G_serving_cell",
    ) if "beam_type" in phone_5g.columns else (
        phone_5g.copy(),
        "Vienna_phone_5G",
    )

    lte_common = pd.DataFrame({
        "source": "Vienna_phone_LTE",
        "network_type": "4G",
        "rsrp_dbm": pd.to_numeric(phone_lte[lte_rsrp_col], errors="coerce"),
        "download_mbps": pd.to_numeric(phone_lte[lte_dl_col], errors="coerce"),
        "upload_mbps": pd.to_numeric(phone_lte[lte_ul_col], errors="coerce"),
        "latency_ms": np.nan,
        "jitter_ms": np.nan,
    })

    fiveg_common = pd.DataFrame({
        "source": fiveg_source,
        "network_type": "5G",
        "rsrp_dbm": pd.to_numeric(phone_5g_used[fiveg_rsrp_col], errors="coerce"),
        "download_mbps": pd.to_numeric(phone_5g_used[fiveg_dl_col], errors="coerce"),
        "upload_mbps": pd.to_numeric(phone_5g_used[fiveg_ul_col], errors="coerce"),
        "latency_ms": np.nan,
        "jitter_ms": np.nan,
    })
    return ensure_common_schema(pd.concat([lte_common, fiveg_common], ignore_index=True))

# Map Vienna scanner files into the common schema. Scanner data is treated as supplementary RSRP-only evidence.
def align_vienna_scanner(scanner_lte: pd.DataFrame | None = None, scanner_5g: pd.DataFrame | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if scanner_lte is not None and not scanner_lte.empty: frames.append(pd.DataFrame({"source": "Vienna_scanner_lte", "network_type": "4G", "rsrp_dbm": pd.to_numeric(scanner_lte[first_existing_column(scanner_lte, ["rsrp_dbm", "rsrp", "RSRP"], "Vienna scanner LTE", "RSRP")], errors="coerce"), "download_mbps": np.nan, "upload_mbps": np.nan, "latency_ms": np.nan, "jitter_ms": np.nan}))
    if scanner_5g is not None and not scanner_5g.empty: frames.append(pd.DataFrame({"source": "Vienna_scanner_5g", "network_type": "5G", "rsrp_dbm": pd.to_numeric(scanner_5g[first_existing_column(scanner_5g, ["rsrp_dbm", "rsrp", "RSRP"], "Vienna scanner 5G", "RSRP")], errors="coerce"), "download_mbps": np.nan, "upload_mbps": np.nan, "latency_ms": np.nan, "jitter_ms": np.nan}))
    return ensure_common_schema(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(columns=COMMON_EXTERNAL_COLUMNS)

# Map Campus QoS throughput tables into the common schema.
def align_campus_qos(campus_tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_name, df in campus_tables.items():
        if df.empty: continue
        frames.append(pd.DataFrame({"source": source_name, "network_type": "5G", "rsrp_dbm": np.nan, "download_mbps": pd.to_numeric(df[first_existing_column(df, ["mbpsactual_downlink", "download_mbps", "dl_mbps"], source_name, "download throughput")], errors="coerce"), "upload_mbps": pd.to_numeric(df[first_existing_column(df, ["mbpsactual_uplink", "upload_mbps", "ul_mbps"], source_name, "upload throughput")], errors="coerce"), "latency_ms": np.nan, "jitter_ms": pd.to_numeric(df[first_existing_column(df, ["meanjitterms_downlink", "jitter_ms", "jitter_downlink_ms"], source_name, "downlink jitter")], errors="coerce")}))
    return ensure_common_schema(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(columns=COMMON_EXTERNAL_COLUMNS)

# Return descriptive statistics required for reporting.
def summary_stats(series: pd.Series) -> dict[str, float | int]:
    x = clean_numeric(series)
    if len(x) == 0: return {"n": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "p05": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "p95": np.nan, "max": np.nan, "iqr": np.nan}
    p25, p75 = float(x.quantile(0.25)), float(x.quantile(0.75))
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "p05": float(x.quantile(0.05)), "p25": p25, "median": float(x.median()), "p75": p75, "p95": float(x.quantile(0.95)), "max": float(x.max()), "iqr": p75 - p25}

# Create descriptive summaries for multiple aligned datasets.
def diagnostic_summary(datasets: Mapping[str, pd.DataFrame], variables: Sequence[str] = ("rsrp_dbm", "download_mbps", "upload_mbps", "jitter_ms")) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for dataset_name, df in datasets.items():
        for variable in variables:
            if variable not in df.columns: continue
            stats = summary_stats(df[variable])
            stats["dataset"], stats["variable"] = dataset_name, variable
            rows.append(stats)
    return pd.DataFrame(rows).loc[:, list(SUMMARY_COLUMNS)] if rows else pd.DataFrame(columns=SUMMARY_COLUMNS)

# Compute Jensen-Shannon distance between two numeric distributions.
def js_distance_hist(x: pd.Series, y: pd.Series, bins: int = 60) -> float:
    sx, sy = clean_numeric(x), clean_numeric(y)
    if len(sx) == 0 or len(sy) == 0: return np.nan
    low, high = min(float(sx.min()), float(sy.min())), max(float(sx.max()), float(sy.max()))
    if high <= low: return 0.0
    hx, bin_edges = np.histogram(sx, bins=bins, range=(low, high), density=True)
    hy, _ = np.histogram(sy, bins=bin_edges, density=True)
    hx, hy = hx + 1e-12, hy + 1e-12
    return float(jensenshannon(hx / hx.sum(), hy / hy.sum()))

# Return numerator / denominator, guarding zero and missing values.
def safe_ratio(numerator: float, denominator: float) -> float:
    return np.nan if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0 else float(numerator) / float(denominator)

# Compare one synthetic variable against one external variable.
def compare_variable(syn_df: pd.DataFrame, ext_df: pd.DataFrame, syn_col: str, ext_col: str, label: str) -> dict[str, float | int | str]:
    sx, ex = clean_numeric(syn_df[syn_col]), clean_numeric(ext_df[ext_col])
    synthetic_median, external_median = float(sx.median()) if len(sx) else np.nan, float(ex.median()) if len(ex) else np.nan
    synthetic_iqr, external_iqr = float(sx.quantile(0.75) - sx.quantile(0.25)) if len(sx) else np.nan, float(ex.quantile(0.75) - ex.quantile(0.25)) if len(ex) else np.nan
    return {"variable": label, "synthetic_column": syn_col, "external_column": ext_col, "synthetic_n": int(len(sx)), "external_n": int(len(ex)), "synthetic_median": synthetic_median, "external_median": external_median, "median_difference": synthetic_median - external_median if pd.notna(synthetic_median) and pd.notna(external_median) else np.nan, "median_ratio": safe_ratio(synthetic_median, external_median), "synthetic_iqr": synthetic_iqr, "external_iqr": external_iqr, "iqr_ratio": safe_ratio(synthetic_iqr, external_iqr), "ks_stat": float(ks_2samp(sx, ex).statistic) if len(sx) and len(ex) else np.nan, "wasserstein_distance": float(wasserstein_distance(sx, ex)) if len(sx) and len(ex) else np.nan, "js_distance": js_distance_hist(sx, ex) if len(sx) and len(ex) else np.nan}

# Compare one variable between matched synthetic/external network types.
def compare_by_network_type(
    syn_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    variable: str,
    label: str,
    network_pairs: Sequence[tuple[str, str]] = (("4G", "4G"), ("5G NSA", "5G")),
    comparison_group: str = "Vienna phone",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for syn_nt, ext_nt in network_pairs:
        row = compare_variable(syn_df[syn_df["network_type"] == syn_nt], ext_df[ext_df["network_type"] == ext_nt], variable, variable, f"{label} ({syn_nt})")
        row["comparison_group"], row["synthetic_network_type"], row["external_network_type"] = comparison_group, syn_nt, ext_nt
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.loc[:, list(COMPARISON_COLUMNS)] if not out.empty else pd.DataFrame(columns=COMPARISON_COLUMNS)

# Create a binned median trend table for supplementary diagnostics.
def binned_median_table(df: pd.DataFrame, x_col: str, y_col: str, source_col: str, bins: int = 10) -> pd.DataFrame:
    temp = df[[x_col, y_col, source_col]].dropna().copy()
    if temp.empty: return pd.DataFrame(columns=[source_col, "bin", "x_mid", "y_med", "y_q25", "y_q75", "n"])
    temp["bin"] = pd.qcut(temp[x_col], q=bins, duplicates="drop")
    out = temp.groupby([source_col, "bin"], observed=True).agg(x_mid=(x_col, "median"), y_med=(y_col, "median"), y_q25=(y_col, lambda s: s.quantile(0.25)), y_q75=(y_col, lambda s: s.quantile(0.75)), n=(y_col, "size")).reset_index()
    out["bin"] = out["bin"].astype(str)
    return out

# Compute source-specific Spearman association summaries.
def spearman_report(df: pd.DataFrame, x_col: str, y_col: str, source_col: str, min_n: int = 10) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for source, group in df.dropna(subset=[x_col, y_col]).groupby(source_col):
        if len(group) < min_n: continue
        rho, p_value = spearmanr(group[x_col], group[y_col])
        rows.append({"source": source, "x": x_col, "y": y_col, "n": int(len(group)), "spearman_rho": float(rho), "p_value": float(p_value)})
    return pd.DataFrame(rows, columns=["source", "x", "y", "n", "spearman_rho", "p_value"])

# Create cautious interpretation flags for external-alignment reporting. This is not generator calibration. It only summarizes where the synthetic median is noticeably high/low relative to the external reference.
def alignment_interpretation_flags(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in comparison_df.iterrows():
        variable, group, median_difference, median_ratio = str(row.get("variable", "")), str(row.get("comparison_group", "")), row.get("median_difference"), row.get("median_ratio")
        if pd.isna(median_difference) and pd.isna(median_ratio): continue
        variable_lower = variable.lower()
        if "rsrp" in variable_lower:
            if pd.notna(median_difference) and float(median_difference) > 8: rows.append({"comparison_group": group, "variable": variable, "flag": "synthetic_stronger_rsrp", "interpretation": "Synthetic RSRP median is stronger than the external reference by more than 8 dB."})
            elif pd.notna(median_difference) and float(median_difference) < -8: rows.append({"comparison_group": group, "variable": variable, "flag": "synthetic_weaker_rsrp", "interpretation": "Synthetic RSRP median is weaker than the external reference by more than 8 dB."})
        if "download" in variable_lower or "upload" in variable_lower:
            if pd.notna(median_ratio) and float(median_ratio) > 1.3: rows.append({"comparison_group": group, "variable": variable, "flag": "synthetic_high_throughput", "interpretation": "Synthetic throughput median is higher than the external reference."})
            elif pd.notna(median_ratio) and float(median_ratio) < 0.7: rows.append({"comparison_group": group, "variable": variable, "flag": "synthetic_low_throughput", "interpretation": "Synthetic throughput median is lower than the external reference."})
        if "jitter" in variable_lower:
            if pd.notna(median_ratio) and float(median_ratio) > 1.3: rows.append({"comparison_group": group, "variable": variable, "flag": "synthetic_high_jitter", "interpretation": "Synthetic jitter median is higher than the external reference."})
            elif pd.notna(median_ratio) and float(median_ratio) < 0.7: rows.append({"comparison_group": group, "variable": variable, "flag": "synthetic_low_jitter", "interpretation": "Synthetic jitter median is lower than the external reference."})
    return pd.DataFrame(rows, columns=["comparison_group", "variable", "flag", "interpretation"])

# Return the external feature mapping table for paper/repo reporting.
def external_feature_mapping_table() -> pd.DataFrame:
    return pd.DataFrame([{"synthetic_variable": "Signal_Strength_dBm", "common_variable": "rsrp_dbm", "external_source": "Vienna phone LTE/5G", "external_columns": "rsrp_dbm", "main_use": "main", "notes": "Primary external radio-signal alignment variable."}, {"synthetic_variable": "Download_Speed_Mbps", "common_variable": "download_mbps", "external_source": "Vienna phone LTE/5G; Campus QoS", "external_columns": "dl_throughput_mbps; mbpsactual_downlink", "main_use": "main", "notes": "Primary QoS distribution-alignment variable."}, {"synthetic_variable": "Jitter_ms", "common_variable": "jitter_ms", "external_source": "Campus QoS", "external_columns": "meanjitterms_downlink", "main_use": "main", "notes": "Used only in controlled 5G-like comparison."}, {"synthetic_variable": "Upload_Speed_Mbps", "common_variable": "upload_mbps", "external_source": "Vienna phone LTE/5G; Campus QoS", "external_columns": "ul_throughput_mbps; mbpsactual_uplink", "main_use": "supplementary/table", "notes": "Secondary QoS check; not required as a main figure."}, {"synthetic_variable": "Signal_Strength_dBm", "common_variable": "rsrp_dbm", "external_source": "Vienna scanner LTE/5G", "external_columns": "rsrp_dbm", "main_use": "supplementary", "notes": "Scanner data is RSRP-only and should not be mixed with connected-UE throughput comparisons."}, {"synthetic_variable": "Packet_Loss", "common_variable": "packet_loss", "external_source": "Campus QoS", "external_columns": "meanloss_downlink; meanloss_uplink", "main_use": "ignored", "notes": "Skip unless SynNetQoS explicitly generates packet-loss variables."}])

# Return the VoNR latency Welch t-test result without plotting.
def check_vonr_latency_consistency(df: pd.DataFrame) -> dict[str, float | int]:
    required = ["Network_Type", "App_Type", "VoNR_Enabled", "Latency_ms"]
    require_columns(df, required, "SynNetQoS VoNR latency consistency check")
    subset = df[(df["Network_Type"] == "5G SA") & (df["App_Type"] == "Call")]
    lat_vonr, lat_non_vonr = clean_numeric(subset[subset["VoNR_Enabled"] == True]["Latency_ms"]), clean_numeric(subset[subset["VoNR_Enabled"] == False]["Latency_ms"])
    if len(lat_vonr) == 0 or len(lat_non_vonr) == 0: return {"vonr_enabled_n": int(len(lat_vonr)), "vonr_disabled_n": int(len(lat_non_vonr)), "t_stat": np.nan, "p_value": np.nan}
    t_stat, p_value = ttest_ind(lat_vonr, lat_non_vonr, equal_var=False, nan_policy="omit")
    return {"vonr_enabled_n": int(len(lat_vonr)), "vonr_disabled_n": int(len(lat_non_vonr)), "t_stat": float(t_stat), "p_value": float(p_value)}
