# SynNetQoS

**SynNetQoS: A Transparent Simulation-Based Synthetic 4G/5G Dataset Generator for QoS and QoE Modeling**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20172326.svg)](https://doi.org/10.5281/zenodo.20172326)

SynNetQoS is a transparent simulation-based synthetic 4G/5G QoS/QoE data-generation framework with external-alignment checks, leakage-aware machine-learning benchmarks, and a controlled KPI-level 5G-LENA/ns-3 simulator-reference comparison.

The repository is organized around six reproducible workflow stages:

1. Synthetic dataset generation
2. Internal consistency checks
3. Monte Carlo stability analysis
4. External alignment against selected public measurement datasets
5. Leakage-aware supervised machine-learning benchmarking
6. Controlled KPI-level 5G-LENA/ns-3 simulator-reference comparison

The public dataset uses anonymized deployment areas, anonymized operator profiles, and UE capability profiles. It should not be interpreted as an empirical ranking of real cities, operators, devices, or network deployments.

The external-alignment workflow is a post-generation distributional check against selected public measurement datasets. It is not a calibration procedure and should not be interpreted as proof that SynNetQoS reproduces full real-world deployment distributions.

The simulator-comparison workflow is a controlled KPI-level comparison against selected 5G-LENA/ns-3 traces. It is intended as simulator-reference evidence for selected aggregate trends, not as packet-level equivalence, validation, or calibration.

## Citation

If you use this software, synthetic dataset, generated result summaries, figures, tables, or CSV files, please cite the archived Zenodo release:

Mohammed Mostafa, Tanvir Alam Tanim, Mst. Asmaul Husna Mayad, Faiza Binte Zaman, & Mohaimen-Bin-Noor. (2026). *SynNetQoS: A Transparent Simulation-Based Synthetic 4G/5G Dataset Generator for QoS and QoE Modeling* (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.20172326

A manuscript preprint describing SynNetQoS is available at SSRN and may be cited as:

Mostafa, Mohammed and Tanim, Tanvir Alam and Mayad, Mst. Asmaul Husna and Binte Zaman, Faiza and Noor, Mohaimen-Bin, SynNetQoS: A Transparent Simulation-Based Synthetic 4G/5G Dataset Generator for QoS and QoE Modeling. Available at SSRN: https://ssrn.com/abstract=6880457 or http://dx.doi.org/10.2139/ssrn.6880457

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
python scripts/06_simulator_comparison.py
pytest
```

## Repository layout
```
synnetqos
├─ CITATION.cff
├─ LICENSE
├─ LICENSE-DATA
├─ NOTICE
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
│  │  ├─ ns3_lena
│  │  │  ├─ processed
│  │  │  │  ├─ ns3_lena_kpis_normalized.csv
│  │  │  │  ├─ simulator_comparison_combined_schema.csv
│  │  │  │  └─ synnetqos_simulator_comparable_subset.csv
│  │  │  └─ raw
│  │  │     ├─ cttc_nr_demo_*_load_seed_*.console.txt
│  │  │     ├─ cttc_nr_demo_*_load_seed_*.txt
│  │  │     ├─ cttc_nr_demo_print_help.txt
│  │  │     └─ ns3_lena_run_manifest.csv
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
│  ├─ simulator_comparison
│  │  └─ simulator_throughput_comparison.pdf
│  └─ supplementary
│     ├─ correlation_heatmap.pdf
│     ├─ dataset_coverage_heatmap.pdf
│     ├─ external_rsrp_to_throughput_trend.pdf
│     ├─ internal_check_mobility_vs_handovers.pdf
│     ├─ internal_check_obstruction_vs_signal.pdf
│     ├─ latency_vonr_boxplot.pdf
│     ├─ ml_future_drop_precision_recall_curves.pdf
│     ├─ ml_streaming_qoe_precision_recall_curves.pdf
│     ├─ monte_carlo_stability.pdf
│     └─ simulator_delay_jitter_comparison.pdf
├─ pyproject.toml
├─ requirements.txt
├─ results
│  ├─ README.md
│  ├─ external_alignment
│  │  ├─ controlled_5g_subset_audit.csv
│  │  ├─ external_alignment_campus_controlled_5g_summary.csv
│  │  ├─ external_alignment_diagnostic_summary.csv
│  │  ├─ external_alignment_interpretation_flags.csv
│  │  ├─ external_alignment_summary.csv
│  │  ├─ external_alignment_vienna_phone_summary.csv
│  │  ├─ external_feature_mapping.csv
│  │  └─ supplementary
│  │     ├─ external_alignment_vienna_positive_downlink_summary.csv
│  │     ├─ external_rsrp_to_throughput_trend.csv
│  │     ├─ external_spearman_rsrp_throughput.csv
│  │     └─ vienna_scanner_rsrp_summary.csv
│  ├─ generator
│  │  ├─ dataset_integrity_summary.csv
│  │  ├─ dataset_schema.csv
│  │  ├─ drop_event_summary.csv
│  │  ├─ generator_config.json
│  │  ├─ monte_carlo_run_metrics.csv
│  │  ├─ monte_carlo_stability_summary.csv
│  │  ├─ numerical_range_summary.csv
│  │  ├─ outlier_report.csv
│  │  ├─ propagation_model_audit.csv
│  │  └─ vonr_latency_consistency.csv
│  ├─ ml_benchmark
│  │  ├─ ml_benchmark_run_metrics.csv
│  │  ├─ ml_benchmark_summary.csv
│  │  ├─ ml_confusion_summary.csv
│  │  ├─ ml_feature_importance.csv
│  │  ├─ ml_feature_sets.csv
│  │  ├─ ml_leakage_audit.csv
│  │  ├─ ml_reproducibility_metadata.json
│  │  ├─ ml_split_summary.csv
│  │  ├─ ml_target_prevalence.csv
│  │  └─ ml_task_definitions.csv
│  └─ simulator_comparison
│     ├─ simulator_comparison_interpretation_flags.csv
│     ├─ simulator_comparison_verdict.csv
│     ├─ simulator_feature_mapping.csv
│     ├─ simulator_kpi_comparison.csv
│     ├─ simulator_kpi_summary.csv
│     ├─ simulator_kpi_trend_summary.csv
│     ├─ simulator_packet_loss_summary.csv
│     ├─ simulator_trace_manifest.json
│     └─ simulator_trace_quality_flags.csv
├─ scripts
│  ├─ 01_generate_dataset.py
│  ├─ 02_internal_consistency_checks.py
│  ├─ 03_monte_carlo_stability.py
│  ├─ 04_external_alignment.py
│  ├─ 05_ml_benchmark.py
│  └─ 06_simulator_comparison.py
├─ simulators
│  └─ ns3_lena
│     └─ run_cttc_nr_demo_grid.sh
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
│     ├─ simulator_comparison.py
│     └─ validation.py
└─ tests
   ├─ test_generator_kpi_scaling.py
   ├─ test_integrity.py
   └─ test_simulator_comparison.py

```

## Data policy

The generated SynNetQoS dataset is stored under `data/synthetic/`.

External measurement datasets and raw simulator traces are treated as local input data. They may be placed under `data/external/` when running the external-alignment and simulator-comparison workflows, but they are not required to be tracked in Git. The reproducible summaries, figures, and comparison outputs are written under `results/` and `figures/`.

Typical local external-input structure:

```text
data/external
├─ campus_qos
├─ ns3_lena
│  ├─ processed
│  └─ raw
└─ vienna
   ├─ phone
   └─ scanner
```

## Workflow outputs

### Dataset generation and internal checks

The generator creates the synthetic dataset and records reproducibility metadata, including the generator configuration and dataset SHA-256 hash. Internal checks summarize dataset integrity, numerical ranges, outlier diagnostics, propagation-model audit values, drop-event behavior, and VoNR latency consistency.

Main outputs are written to:

```text
data/synthetic/
results/generator/
figures/supplementary/
```

### External alignment

The external-alignment workflow compares selected SynNetQoS variables with Vienna phone measurements and Campus QoS throughput data. The main variables are RSRP, download throughput, and controlled 5G jitter where comparable external references are available.

Main outputs are written to:

```text
results/external_alignment/
figures/external_alignment/
```

Supplementary external diagnostics are written to:

```text
results/external_alignment/supplementary/
figures/supplementary/
```

These outputs should be interpreted as selected distributional sanity checks, not as full real-world calibration.

### Machine-learning benchmark

The machine-learning benchmark evaluates structured learnability under leakage-aware synthetic benchmark settings. The benchmark uses session-wise grouped splitting, validation-set threshold selection, and repeated random seeds. The benchmark repeats the split-and-train procedure across three fixed seeds, records per-run metrics, and writes split summaries to support reproducibility.

Main outputs are written to:

```text
results/ml_benchmark/
figures/ml_benchmark/
```

Supplementary precision-recall curves are written to:

```text
figures/supplementary/
```

The benchmark should be interpreted as evidence of learnability under controlled synthetic-data conditions. It is not evidence of real-world deployment performance.

### Simulator comparison

The simulator-comparison workflow compares selected aggregate KPIs from SynNetQoS against controlled 5G-LENA/ns-3 traces generated with the `cttc-nr-demo` setup across low, medium, and high offered-load settings. The simulator trace manifest is expected to contain 60 simulator runs: 20 seeds for each of the low, medium, and high offered-load settings.

Main outputs are written to:

```text
results/simulator_comparison/
figures/simulator_comparison/
figures/supplementary/
```

The throughput comparison is the main simulator-reference figure. Delay and jitter are retained as diagnostic outputs because their measurement semantics differ between a session-level synthetic generator and packet-level simulator traces. Packet loss is reported only as a simulator-side diagnostic unless a comparable SynNetQoS packet-loss field is introduced in a future workflow.

## Interpretation boundaries

SynNetQoS is designed for transparent synthetic-data generation, auditability, external-alignment checking, and leakage-aware benchmarking. The current repository supports these claims:

- the dataset is generated from a fixed, inspectable configuration;
- generated variables can be audited through reproducibility and integrity checks;
- selected radio and QoS/QoE variables can be compared against external references;
- leakage-aware ML benchmarks can be run with session-wise grouped splits;
- selected aggregate simulator KPIs can be compared against 5G-LENA/ns-3 traces as a controlled reference check.

The current repository does not claim:

- field calibration against operational network deployments;
- exact reproduction of real-world measurement distributions;
- packet-level equivalence with 5G-LENA/ns-3;
- real-world predictive deployment performance;
- empirical ranking of real cities, operators, devices, or network deployments.

## License

The source code in this repository is licensed under the Apache License 2.0.

The synthetic dataset and research outputs, including generated result summaries, figures, tables, and CSV files, are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0), unless otherwise stated.
