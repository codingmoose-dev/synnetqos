from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import lognorm, norm

# Apply plotting style.
def setup_plot_style() -> None:
    figure_palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#6E6E6E"]
    sns.set_theme(style="ticks", palette=figure_palette, rc={"figure.figsize": (8, 5), "figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "font.size": 10, "lines.linewidth": 1.8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})
    plt.rcParams["pdf.fonttype"], plt.rcParams["ps.fonttype"] = 42, 42

# Convert a series to finite numeric values only.
def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

# Plot a correlation heatmap for selected synthetic QoS variables.
def plot_correlation_heatmap(df: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(df[features].corr(), cmap=sns.diverging_palette(230, 20, as_cmap=True), center=0, annot=True, fmt=".2f", annot_kws={"size": 8}, square=True, linewidths=0.5, cbar_kws={"shrink": 0.75}, ax=ax)
    # LaTeX caption/title candidate: "Correlation among Primary Synthetic QoS Parameters"
    # ax.set_title("Primary Synthetic QoS Parameters Correlation", fontsize=20, weight="bold", pad=20)
    plt.setp(ax.get_xticklabels(), fontsize=12, rotation=45, ha="right")
    fig.tight_layout(pad=1.5)
    return fig

# Plot coverage of network type by movement speed.
def plot_scenario_coverage(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pd.crosstab(df["Network_Type"], df["Movement_Speed"]), annot=True, fmt="d", cmap="viridis", linewidths=0.5, ax=ax)
    # LaTeX caption/title candidate: "Dataset Coverage by Network Type and Movement Speed"
    # ax.set_title("Dataset Coverage: Network Type vs Movement Speed")
    ax.set_xlabel("Movement Speed")
    ax.set_ylabel("Network Type")
    fig.tight_layout()
    return fig

# Plot whether higher mobility corresponds to more handovers.
def plot_mobility_vs_handovers(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="Movement_Speed", y="Interval_Handover_Count", order=["Static", "Walking", "Driving"], ax=ax)
    # LaTeX caption/title candidate: "Mobility-Related Variation in Handover Count"
    # ax.set_title("Internal Consistency: Handovers vs Mobility")
    ax.set_xlabel("Movement Speed")
    ax.set_ylabel("Number of Handovers")
    fig.tight_layout()
    return fig

# Plot whether stronger obstruction corresponds to weaker signal.
def plot_obstruction_vs_signal(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="Obstruction_Level", y="Signal_Strength_dBm", order=["Low", "Medium", "High"], ax=ax)
    # LaTeX caption/title candidate: "Signal Strength Variation across Obstruction Levels"
    # ax.set_title("Internal Consistency: Signal Strength vs Obstruction")
    ax.set_xlabel("Obstruction Level")
    ax.set_ylabel("Signal strength (dBm)")
    fig.tight_layout()
    return fig

# Plot latency by VoNR status for internal consistency reporting.
def plot_vonr_latency_boxplot(df: pd.DataFrame) -> plt.Figure:
    subset = df[(df["Network_Type"] == "5G SA") & (df["App_Type"] == "Call")].copy()
    subset["VoNR_Status"] = subset["VoNR_Enabled"].map({False: "Disabled", True: "Enabled"})
    subset = subset.dropna(subset=["VoNR_Status", "Latency_ms"])

    fig, ax = plt.subplots(figsize=(8, 5))

    if subset.empty:
        ax.text(
            0.5,
            0.5,
            "No 5G SA call sessions available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        sns.boxplot(
            data=subset,
            x="VoNR_Status",
            y="Latency_ms",
            order=["Disabled", "Enabled"],
            ax=ax,
        )

    # LaTeX caption/title candidate: "Latency by VoNR Status in 5G SA Call Sessions"
    # ax.set_title("Latency by VoNR Status in 5G SA Call Sessions")
    ax.set_xlabel("VoNR status")
    ax.set_ylabel("Latency (ms)")
    fig.tight_layout()
    return fig

# Archived distribution-fit plots. These figures are exploratory diagnostics only. Do not call this function from publication-facing scripts unless the paper explicitly discusses these parametric fits.
def plot_distributions(df: pd.DataFrame) -> dict[str, plt.Figure]:
    figures: dict[str, plt.Figure] = {}
    latency = _clean_numeric(df["Latency_ms"])
    if len(latency) > 0:
        shape, loc, scale = lognorm.fit(latency, floc=0)
        x_vals, (fig_lat, ax_lat) = np.linspace(latency.min(), latency.max(), 200), plt.subplots()
        sns.histplot(latency, bins=50, stat="density", label="Empirical data", ax=ax_lat)
        ax_lat.plot(x_vals, lognorm.pdf(x_vals, shape, loc, scale), linestyle="--", linewidth=2, label=f"Log-normal fit (shape={shape:.2f})")
        # LaTeX caption/title candidate: "Latency Distribution"
        # ax_lat.set_title("Distribution of Latency")
        ax_lat.set_xlabel("Latency (ms)")
        ax_lat.set_ylabel("Density")
        ax_lat.legend()
        fig_lat.tight_layout()
        figures["latency"] = fig_lat
    jitter = _clean_numeric(df["Jitter_ms"])
    if len(jitter) > 0:
        mu, std = norm.fit(jitter)
        x_vals_j, (fig_jit, ax_jit) = np.linspace(jitter.min(), jitter.max(), 200), plt.subplots()
        sns.histplot(jitter, bins=50, stat="density", label="Empirical data", ax=ax_jit)
        ax_jit.plot(x_vals_j, norm.pdf(x_vals_j, mu, std), linestyle="--", linewidth=2, label=f"Normal fit ($\\mu$={mu:.2f}, $\\sigma$={std:.2f})")
        # LaTeX caption/title candidate: "Jitter Distribution"
        # ax_jit.set_title("Distribution of Jitter")
        ax_jit.set_xlabel("Jitter (ms)")
        ax_jit.set_ylabel("Density")
        ax_jit.legend()
        fig_jit.tight_layout()
        figures["jitter"] = fig_jit
    return figures

# Plot run-level Monte Carlo stability using relative deviations.
def plot_monte_carlo_stability(
    stats_df: pd.DataFrame,
    title: str = "Monte Carlo Stability across Independent Generator Runs",
) -> plt.Figure:
    numeric_df = stats_df.select_dtypes(include=[np.number]).copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    if numeric_df.empty:
        # LaTeX caption/title candidate: title
        # ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No Monte Carlo metrics available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_xlabel("")
        ax.set_ylabel("Deviation from across-run mean (%)")
        fig.tight_layout()
        return fig

    means = numeric_df.mean().replace(0, np.nan)

    plot_df = (
        ((numeric_df - means) / means.abs() * 100)
        .reset_index(names="run")
        .melt(
            id_vars="run",
            var_name="Metric",
            value_name="Deviation from across-run mean (%)",
        )
    )
    plot_df["run"] = plot_df["run"] + 1
    plot_df = plot_df.dropna(subset=["Deviation from across-run mean (%)"])

    if plot_df.empty:
        ax.text(
            0.5,
            0.5,
            "No finite Monte Carlo deviations available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        sns.boxplot(
            data=plot_df,
            x="Metric",
            y="Deviation from across-run mean (%)",
            ax=ax,
        )
        ax.axhline(0, linestyle="--", linewidth=1)

    # LaTeX caption/title candidate: title
    # ax.set_title(title)

    ax.set_xlabel("")
    ax.set_ylabel("Deviation from across-run mean (%)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig

# Plot an ECDF comparison between one synthetic and one external series.
def plot_ecdf(syn_series: pd.Series, ext_series: pd.Series, title: str, xlabel: str, syn_label: str = "SynNetQoS", ext_label: str = "External") -> plt.Figure:
    syn_values, ext_values, (fig, ax) = np.sort(_clean_numeric(syn_series)), np.sort(_clean_numeric(ext_series)), plt.subplots(figsize=(8, 5))
    if len(syn_values): ax.plot(syn_values, np.arange(1, len(syn_values) + 1) / len(syn_values), label=f"{syn_label} (n={len(syn_values):,})")
    if len(ext_values): ax.plot(ext_values, np.arange(1, len(ext_values) + 1) / len(ext_values), linestyle="--", label=f"{ext_label} (n={len(ext_values):,})")
    # ax.set_title(title)
    ax.set_xlabel(xlabel), ax.set_ylabel("ECDF"), ax.legend(), fig.tight_layout()
    return fig

# Plot grouped ECDFs for matched synthetic/external network types.
def plot_ecdf_by_group(
    syn_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    variable: str,
    title: str,
    xlabel: str,
    group_pairs: list[tuple[str, str]] | None = None,
    syn_label_prefix: str = "SynNetQoS",
    ext_label_prefix: str = "External",
    syn_display_names: dict[str, str] | None = None,
    ext_display_names: dict[str, str] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))

    syn_display_names = syn_display_names or {}
    ext_display_names = ext_display_names or {}

    for syn_network, ext_network in group_pairs or [("4G", "4G"), ("5G NSA", "5G")]:
        syn_values = np.sort(_clean_numeric(syn_df.loc[syn_df["network_type"] == syn_network, variable]))
        ext_values = np.sort(_clean_numeric(ext_df.loc[ext_df["network_type"] == ext_network, variable]))

        syn_label = syn_display_names.get(syn_network, syn_network)
        ext_label = ext_display_names.get(ext_network, ext_network)

        if len(syn_values):
            ax.plot(
                syn_values,
                np.arange(1, len(syn_values) + 1) / len(syn_values),
                label=f"{syn_label_prefix} {syn_label} (n={len(syn_values):,})",
            )

        if len(ext_values):
            ax.plot(
                ext_values,
                np.arange(1, len(ext_values) + 1) / len(ext_values),
                linestyle="--",
                label=f"{ext_label_prefix} {ext_label} (n={len(ext_values):,})",
            )

    # LaTeX caption/title candidate: title
    # ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical cumulative probability")
    ax.legend()
    fig.tight_layout()
    return fig

# Plot a supplementary binned median trend.
def plot_binned_median_trend(trend_df: pd.DataFrame, title: str, xlabel: str, ylabel: str, source_col: str = "source") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    if trend_df.empty:
        # ax.set_title(title)
        ax.set_xlabel(xlabel), ax.set_ylabel(ylabel), ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center", va="center"), fig.tight_layout()
        return fig
    for source, group in trend_df.groupby(source_col):
        group = group.sort_values("x_mid")
        ax.plot(group["x_mid"], group["y_med"], marker="o", label=str(source))
    # ax.set_title(title)
    ax.set_xlabel(xlabel), ax.set_ylabel(ylabel), ax.legend(), fig.tight_layout()
    return fig

# Plot mean ML benchmark performance across tasks and models.
def plot_ml_task_result_bars(
    summary_df: pd.DataFrame,
    metric: str = "average_precision",
) -> plt.Figure:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    required = ["task_label", "model", mean_col]
    missing = [col for col in required if col not in summary_df.columns]

    if missing:
        raise KeyError(f"summary_df is missing required columns for ML bar plot: {missing}")

    plot_df = summary_df.copy()

    if std_col not in plot_df.columns:
        plot_df[std_col] = 0.0

    plot_df["task_model"] = (
        plot_df["task_label"].astype(str) + " — " + plot_df["model"].astype(str)
    )

    plot_df = plot_df.sort_values(
        ["task_label", mean_col],
        ascending=[True, True],
    )

    fig_height = max(4.8, 0.32 * len(plot_df))
    fig, ax = plt.subplots(figsize=(8, fig_height))

    positions = np.arange(len(plot_df))

    ax.barh(
        positions,
        plot_df[mean_col],
        xerr=plot_df[std_col].fillna(0.0),
        capsize=3,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(plot_df["task_model"])
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    return fig

# Plot precision-recall curves for one selected benchmark task.
def plot_ml_precision_recall_curves(
    curve_rows: list[dict[str, object]],
    task_id: str = "C_future_drop_context_only",
    title: str = "Precision-Recall Curves: Leakage-Controlled Future-Drop Prediction",
) -> plt.Figure:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    selected = [row for row in curve_rows if row.get("task_id") == task_id]

    if not selected and curve_rows:
        selected = [
            row
            for row in curve_rows
            if row.get("task_id") == curve_rows[0].get("task_id")
        ]

    fig, ax = plt.subplots(figsize=(8, 5))

    if not selected:
        ax.text(
            0.5,
            0.5,
            "No curve data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        fig.tight_layout()
        return fig

    base_rate = None

    for row in selected:
        y_true = np.asarray(row["y_true"])
        y_prob = np.asarray(row["y_prob"])

        if len(np.unique(y_true)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        model = str(row.get("model", "model"))

        ax.plot(
            recall,
            precision,
            label=f"{model} (AP={ap:.3f})",
        )

        base_rate = float(np.mean(y_true))

    if base_rate is not None:
        ax.axhline(
            base_rate,
            linestyle="--",
            linewidth=1,
            label=f"Base rate={base_rate:.3f}",
        )

    # LaTeX caption/title candidate: title
    # ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    return fig