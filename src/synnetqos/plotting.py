from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_correlation_heatmap(df: pd.DataFrame, columns: list[str], outpath: str | Path) -> None:
    """Optional supplementary correlation plot.

    Keep this as supplementary unless the paper directly discusses it.
    """
    import seaborn as sns

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    corr = df[columns].corr()
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        corr,
        cmap=sns.diverging_palette(230, 20, as_cmap=True),
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.75},
    )
    plt.title("Correlation heatmap of primary synthetic QoS/QoE variables")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()
