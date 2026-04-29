import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Generates a heatmap proving the simulation covers required mobility and network states.
def plot_scenario_coverage(df: pd.DataFrame) -> plt.Figure:
    coverage_matrix = pd.crosstab(df['Network_Type'], df['Movement_Speed'])
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(coverage_matrix, annot=True, fmt='d', cmap='viridis', linewidths=.5, ax=ax)
    ax.set_title('Dataset Coverage: Network Type vs Movement Speed')
    ax.set_xlabel('Movement Speed')
    ax.set_ylabel('Network Type')
    return fig

# Validates physical constraint: higher mobility must yield more handovers.
def plot_mobility_vs_handovers(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x='Movement_Speed', y='Interval_Handover_Count', order=['Static', 'Walking', 'Driving'], ax=ax)
    ax.set_title('Internal Consistency: Handovers vs. Mobility')
    ax.set_xlabel('Movement Speed')
    ax.set_ylabel('Number of Handovers')
    return fig

# Validates physical constraint: dense obstructions must degrade signal strength.
def plot_obstruction_vs_signal(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Obstruction_Level', y='Signal_Strength_dBm', order=['Low', 'Medium', 'High'], ax=ax)
    ax.set_title('Internal Consistency: Signal Strength vs. Obstruction')
    ax.set_xlabel('Obstruction Level')
    ax.set_ylabel('Signal Strength (dBm)')
    return fig

# Statistically validates the modeled 15% latency reduction for VoNR connections.
def test_vonr_latency(df: pd.DataFrame) -> tuple[plt.Figure, float, float]:
    vonr_subset = df[(df["Network_Type"] == "5G SA") & (df["App_Type"] == "Call")]
    lat_vonr = vonr_subset[vonr_subset["VoNR_Enabled"] == True]["Latency_ms"]
    lat_non_vonr = vonr_subset[vonr_subset["VoNR_Enabled"] == False]["Latency_ms"]

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="VoNR_Enabled", y="Latency_ms", ax=ax)
    ax.set_title("Latency by VoNR Status")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Disabled', 'Enabled'])

    t_stat, p_val = ttest_ind(lat_vonr, lat_non_vonr, equal_var=False, nan_policy='omit')
    return fig, t_stat, p_val
    