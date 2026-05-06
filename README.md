# SynNetQoS

SynNetQoS is a transparent simulation-based synthetic 4G/5G QoS/QoE data-generation framework with external-alignment checks and leakage-aware machine-learning benchmarks.

The repository is organized around five reproducible workflow stages:

1. Synthetic dataset generation
2. Internal consistency checks
3. Monte Carlo stability analysis
4. External alignment against selected public measurement datasets
5. Leakage-aware supervised machine-learning benchmarking

The public dataset uses anonymized deployment areas, anonymized operator profiles, and UE capability profiles. It should not be interpreted as an empirical ranking of real cities, operators, devices, or network deployments.

## Quick setup

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Main workflow commands

Run from the repository root:

```bash
python scripts/01_generate_dataset.py
python scripts/02_internal_consistency_checks.py
python scripts/03_monte_carlo_stability.py
python scripts/04_external_alignment.py
python scripts/05_ml_benchmark.py
pytest
```

## Repository Tree

```
synnetqos
├─ README.md
├─ config
│  ├─ generator_reference.yaml
│  └─ paths.example.yaml
├─ data
│  ├─ README.md
│  ├─ external
│  │  ├─ campus_qos
│  │  │  ├─ ntnu_tput_all_Throughput.csv
│  │  │  └─ wue_tput_all_Throughput.csv
│  │  └─ vienna
│  │     ├─ phone
│  │     │  ├─ phone_data_5g.parquet
│  │     │  └─ phone_data_lte.parquet
│  │     └─ scanner
│  │        ├─ scanner_data_5g.parquet
│  │        └─ scanner_data_lte.parquet
│  └─ synthetic
│     └─ synnetqos-dataset.csv
├─ figures
│  ├─ README.md
│  ├─ external_alignment
│  │  ├─ external_download_ecdf_syn_vs_campus_controlled.pdf
│  │  ├─ external_download_ecdf_vienna_phone_matched.pdf
│  │  ├─ external_jitter_ecdf_syn_vs_campus_controlled.pdf
│  │  └─ external_rsrp_ecdf_vienna_phone_matched.pdf
│  ├─ ml_benchmark
│  │  └─ ml_task_result_bars.pdf
│  └─ supplementary
│     ├─ correlation_heatmap.pdf
│     ├─ dataset_coverage_heatmap.pdf
│     ├─ external_rsrp_to_throughput_trend.pdf
│     ├─ internal_check_mobility_vs_handovers.pdf
│     ├─ internal_check_obstruction_vs_signal.pdf
│     ├─ latency_vonr_boxplot.pdf
│     ├─ ml_precision_recall_curves.pdf
│     └─ monte_carlo_stability.pdf
├─ pyproject.toml
├─ requirements.txt
├─ results
│  ├─ README.md
│  ├─ external_alignment
│  │  ├─ controlled_5g_subset_audit.csv
│  │  ├─ external_alignment_interpretation_flags.csv
│  │  ├─ external_alignment_summary.csv
│  │  ├─ external_feature_mapping.csv
│  │  ├─ external_alignment_campus_controlled_5g_summary.csv
│  │  ├─ external_alignment_diagnostic_summary.csv
│  │  ├─ external_alignment_vienna_phone_summary.csv
│  │  └─ supplementary
│  │     ├─ external_rsrp_to_throughput_trend.csv
│  │     ├─ external_spearman_rsrp_throughput.csv
│  │     ├─ external_alignment_vienna_positive_downlink_summary.csv
│  │     └─ vienna_scanner_rsrp_summary.csv
│  ├─ generator
│  │  ├─ dataset_integrity_summary.csv
│  │  ├─ dataset_schema.csv
│  │  ├─ generator_config.json
│  │  ├─ monte_carlo_run_metrics.csv
│  │  ├─ monte_carlo_stability_summary.csv
│  │  ├─ numerical_range_summary.csv
│  │  └─ outlier_report.csv
│  └─ ml_benchmark
│     ├─ ml_benchmark_run_metrics.csv
│     ├─ ml_benchmark_summary.csv
│     ├─ ml_confusion_summary.csv
│     ├─ ml_feature_importance.csv
│     ├─ ml_feature_sets.csv
│     ├─ ml_leakage_audit.csv
│     ├─ ml_reproducibility_metadata.json
│     ├─ ml_split_summary.csv
│     └─ ml_task_definitions.csv
├─ scripts
│  ├─ 01_generate_dataset.py
│  ├─ 02_internal_consistency_checks.py
│  ├─ 03_monte_carlo_stability.py
│  ├─ 04_external_alignment.py
│  └─ 05_ml_benchmark.py
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