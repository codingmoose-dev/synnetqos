import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm, norm

def setup_plot_style():
    sns.set_theme(style="whitegrid", palette="colorblind", rc={"figure.figsize": (10, 6), "axes.titlesize": 16, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "grid.linestyle": '--', "grid.alpha": 0.7})

def plot_correlation_heatmap(df: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(df[features].corr(), cmap=sns.diverging_palette(230, 20, as_cmap=True), center=0, annot=True, fmt=".2f", annot_kws={"size": 8}, square=True, linewidths=.5, cbar_kws={"shrink": .75}, ax=ax)
    ax.set_title("Primary Synthetic QoS Parameters Correlation", fontsize=20, weight='bold', pad=20)
    plt.setp(ax.get_xticklabels(), fontsize=12, rotation=45, ha='right')
    fig.tight_layout(pad=1.5)
    return fig

def plot_distributions(df: pd.DataFrame) -> dict[str, plt.Figure]:
    figures = {}
    
    latency = df["Latency_ms"].dropna()
    shape, loc, scale = lognorm.fit(latency, floc=0)
    x_vals = np.linspace(latency.min(), latency.max(), 200)
    fig_lat, ax_lat = plt.subplots()
    sns.histplot(latency, bins=50, stat='density', label='Empirical Data', ax=ax_lat)
    ax_lat.plot(x_vals, lognorm.pdf(x_vals, shape, loc, scale), 'r--', lw=2, label=f'Log-Normal (shape={shape:.2f})')
    ax_lat.set_title("Distribution of Latency")
    ax_lat.legend()
    figures["latency"] = fig_lat

    jitter = df["Jitter_ms"].dropna()
    mu, std = norm.fit(jitter)
    x_vals_j = np.linspace(jitter.min(), jitter.max(), 200)
    fig_jit, ax_jit = plt.subplots()
    sns.histplot(jitter, bins=50, stat='density', label='Empirical Data', ax=ax_jit)
    ax_jit.plot(x_vals_j, norm.pdf(x_vals_j, mu, std), 'r--', lw=2, label=f'Normal ($\mu$={mu:.2f}, $\sigma$={std:.2f})')
    ax_jit.set_title("Distribution of Jitter")
    ax_jit.legend()
    figures["jitter"] = fig_jit
    
    return figures
    