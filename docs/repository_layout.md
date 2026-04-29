# Repository layout

`src/` means **source code**. It stores reusable Python package code instead of mixing everything inside one notebook. Scripts in `scripts/` call functions from `src/synnetqos/`.

Recommended layout:

```text
SynNetQoS/
├── src/synnetqos/                 # reusable package code
├── scripts/                       # runnable workflows
├── config/                        # reference settings
├── data/                          # local data; raw external data not committed
├── results/                       # generated CSV/JSON outputs
├── figures/                       # paper and supplementary figures
├── notebooks/exploratory/         # optional analysis only
├── docs/                          # notes, schema, dataset card
├── tests/                         # integrity tests
└── archive/                       # old notebooks/scripts, not paper-facing
```

Final paper-facing code should not mix generation, debugging, exploratory plotting, external validation, and ML benchmarking in a single file.
