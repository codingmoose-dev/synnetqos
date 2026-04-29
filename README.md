# SynNetQoS

SynNetQoS is a transparent simulation-based synthetic 4G/5G QoS/QoE data-generation framework and leakage-aware benchmark suite.

The repository is organized around three reproducible workflows:

1. Synthetic dataset generation
2. External alignment against selected public measurement datasets
3. Leakage-aware supervised ML benchmarking

The public dataset should use anonymized deployment areas, anonymized operator profiles, and UE capability profiles. It should not be interpreted as an empirical ranking of real cities, operators, or devices.

## Quick setup

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Main commands

```bash
python scripts/01_generate_dataset.py
python scripts/02_external_alignment.py
python scripts/03_monte_carlo_stability.py
python scripts/04_ml_benchmark.py
```

```
synnetqos_publishable_scaffold
├─ README.md
├─ config
│  ├─ generator_reference.yaml
│  └─ paths.example.yaml
├─ data
│  └─ README.md
├─ docs
│  └─ repository_layout.md
├─ figures
│  └─ README.md
├─ notebooks
│  └─ exploratory
│     └─ README.md
├─ requirements.txt
├─ scripts
│  ├─ 01_generate_dataset.py
│  ├─ 02_external_alignment.py
│  ├─ 03_monte_carlo_stability.py
│  └─ 04_ml_benchmark.py
├─ src
│  └─ synnetqos
│     ├─ __init__.py
│     ├─ audits.py
│     ├─ config.py
│     ├─ generator.py
│     ├─ io.py
│     ├─ ml.py
│     ├─ plotting.py
│     ├─ profiles.py
│     └─ validation.py
└─ tests
   └─ test_integrity.py

```