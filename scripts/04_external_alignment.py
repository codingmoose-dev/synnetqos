# Run external alignment against Vienna and Campus QoS datasets.
# 
# Main outputs:
# - results/external_alignment/external_validation_vienna_phone_summary.csv
# - results/external_alignment/external_validation_campus_controlled_5g_summary.csv
# - results/external_alignment/external_alignment_summary.csv
# - results/external_alignment/external_feature_mapping.csv
# - results/external_alignment/external_validation_diagnostic_summary.csv
# - results/external_alignment/external_alignment_interpretation_flags.csv
# - results/external_alignment/controlled_5g_subset_audit.csv
# - figures/external_alignment/external_rsrp_ecdf_vienna_phone_matched.pdf
# - figures/external_alignment/external_download_ecdf_vienna_phone_matched.pdf
# - figures/external_alignment/external_download_ecdf_syn_vs_campus_controlled.pdf
# - figures/external_alignment/external_jitter_ecdf_syn_vs_campus_controlled.pdf
# 
# Supplementary outputs:
# - results/external_alignment/supplementary/external_validation_vienna_positive_downlink_summary.csv
# - results/external_alignment/supplementary/external_rsrp_to_throughput_trend.csv
# - results/external_alignment/supplementary/external_spearman_rsrp_throughput.csv
# - results/external_alignment/supplementary/vienna_scanner_rsrp_summary.csv
# - figures/supplementary/external_rsrp_to_throughput_trend.pdf

from __future__ import annotations
from pathlib import Path
import pandas as pd

from synnetqos.io import path_exists, read_csv, read_parquet, save_plot, write_csv
from synnetqos.plotting import plot_binned_median_trend, plot_ecdf, plot_ecdf_by_group, setup_plot_style
from synnetqos.validation import align_campus_qos, align_synnetqos, align_vienna_phone, align_vienna_scanner, alignment_interpretation_flags, binned_median_table, build_controlled_5g_synnetqos, compare_by_network_type, compare_variable, diagnostic_summary, external_feature_mapping_table, spearman_report


def main() -> None:
    setup_plot_style()
    data_path = Path("data/synthetic/synnetqos-dataset.csv")
    vienna_phone_dir, vienna_scanner_dir, campus_dir = Path("data/external/vienna/phone"), Path("data/external/vienna/scanner"), Path("data/external/campus_qos")
    vienna_phone_5g_path, vienna_phone_lte_path = vienna_phone_dir / "phone_data_5g.parquet", vienna_phone_dir / "phone_data_lte.parquet"
    vienna_scanner_5g_path, vienna_scanner_lte_path = vienna_scanner_dir / "scanner_data_5g.parquet", vienna_scanner_dir / "scanner_data_lte.parquet"
    campus_ntnu_path, campus_wue_path = campus_dir / "ntnu_tput_all_Throughput.csv", campus_dir / "wue_tput_all_Throughput.csv"
    result_dir, supplementary_result_dir = Path("results/external_alignment"), Path("results/external_alignment") / "supplementary"
    figure_dir, supplementary_figure_dir = Path("figures/external_alignment"), Path("figures/supplementary")

    required_paths = [data_path, vienna_phone_5g_path, vienna_phone_lte_path]
    missing_required = [str(path) for path in required_paths if not path_exists(path)]
    if missing_required: raise FileNotFoundError("Missing required external-alignment inputs:\n" + "\n".join(f"- {path}" for path in missing_required))

    campus_tables: dict[str, pd.DataFrame] = {}
    if path_exists(campus_ntnu_path): campus_tables["Campus_NTNU_5G_QoS"] = read_csv(campus_ntnu_path)
    if path_exists(campus_wue_path): campus_tables["Campus_WUE_5G_QoS"] = read_csv(campus_wue_path)
    if not campus_tables: raise FileNotFoundError(f"No Campus QoS throughput files were found. Expected at least one of:\n- {campus_ntnu_path}\n- {campus_wue_path}")

    print("Loading SynNetQoS and external reference datasets...")
    syn_df = read_csv(data_path)
    vienna_phone_5g, vienna_phone_lte = read_parquet(vienna_phone_5g_path), read_parquet(vienna_phone_lte_path)

    print("Mapping datasets into the common validation schema...")
    syn_common, vienna_phone_common, campus_common = align_synnetqos(syn_df), align_vienna_phone(vienna_phone_lte, vienna_phone_5g), align_campus_qos(campus_tables)
    syn_campus_common, controlled_5g_audit = build_controlled_5g_synnetqos(syn_df)

    write_csv(external_feature_mapping_table(), result_dir / "external_feature_mapping.csv")
    write_csv(controlled_5g_audit, result_dir / "controlled_5g_subset_audit.csv")

    print("Computing Vienna phone external-alignment summaries...")
    vienna_network_pairs = [("4G", "4G"), ("5G NSA", "5G")]

    vienna_rsrp_table = compare_by_network_type(
        syn_common,
        vienna_phone_common,
        variable="rsrp_dbm",
        label="RSRP",
        network_pairs=vienna_network_pairs,
        comparison_group="Vienna phone",
    )

    vienna_downlink_table = compare_by_network_type(
        syn_common,
        vienna_phone_common,
        variable="download_mbps",
        label="Download throughput",
        network_pairs=vienna_network_pairs,
        comparison_group="Vienna phone",
    )

    vienna_uplink_table = compare_by_network_type(
        syn_common,
        vienna_phone_common,
        variable="upload_mbps",
        label="Upload throughput",
        network_pairs=vienna_network_pairs,
        comparison_group="Vienna phone",
    )
    
    vienna_external_summary = pd.concat([vienna_rsrp_table, vienna_downlink_table, vienna_uplink_table], ignore_index=True)
    write_csv(vienna_external_summary, result_dir / "external_validation_vienna_phone_summary.csv")

    print("Computing Campus controlled-5G external-alignment summaries...")
    campus_rows = []
    for column, label in [("download_mbps", "Controlled 5G download throughput"), ("upload_mbps", "Controlled 5G upload throughput"), ("jitter_ms", "Controlled 5G downlink jitter")]:
        row = compare_variable(syn_campus_common, campus_common, column, column, label)
        row["comparison_group"], row["synthetic_network_type"], row["external_network_type"] = "Campus controlled 5G", "controlled 5G subset", "5G"
        campus_rows.append(row)
    campus_external_summary = pd.DataFrame(campus_rows)
    write_csv(campus_external_summary, result_dir / "external_validation_campus_controlled_5g_summary.csv")

    external_alignment_summary = pd.concat([vienna_external_summary, campus_external_summary], ignore_index=True)
    write_csv(external_alignment_summary, result_dir / "external_alignment_summary.csv")

    interpretation_flags = alignment_interpretation_flags(external_alignment_summary)
    write_csv(interpretation_flags, result_dir / "external_alignment_interpretation_flags.csv")

    print("Writing supplementary diagnostic summaries...")
    vienna_positive_downlink_summary = compare_by_network_type(
        syn_common[syn_common["download_mbps"] > 0],
        vienna_phone_common[vienna_phone_common["download_mbps"] > 0],
        variable="download_mbps",
        label="Positive download throughput",
        network_pairs=vienna_network_pairs,
        comparison_group="Vienna phone positive-throughput subset",
    )
    write_csv(vienna_positive_downlink_summary, supplementary_result_dir / "external_validation_vienna_positive_downlink_summary.csv")

    diagnostic_df = diagnostic_summary({
        "SynNetQoS_all": syn_common,
        "SynNetQoS_4G": syn_common[syn_common["network_type"] == "4G"],
        "SynNetQoS_5G_NSA": syn_common[syn_common["network_type"] == "5G NSA"],
        "SynNetQoS_5G_SA": syn_common[syn_common["network_type"] == "5G SA"],
        "SynNetQoS_controlled_5G_subset": syn_campus_common,
        "Vienna_phone_LTE": vienna_phone_common[vienna_phone_common["network_type"] == "4G"],
        "Vienna_phone_5G_serving_cell": vienna_phone_common[vienna_phone_common["network_type"] == "5G"],
        "Campus_5G_QoS": campus_common,
    })
    write_csv(diagnostic_df, result_dir / "external_validation_diagnostic_summary.csv")

    combined_rsrp_throughput = pd.concat([syn_common[["source", "rsrp_dbm", "download_mbps"]], vienna_phone_common[["source", "rsrp_dbm", "download_mbps"]]], ignore_index=True).dropna(subset=["rsrp_dbm", "download_mbps"])
    trend_table = binned_median_table(combined_rsrp_throughput, x_col="rsrp_dbm", y_col="download_mbps", source_col="source", bins=10)
    write_csv(trend_table, supplementary_result_dir / "external_rsrp_to_throughput_trend.csv")

    spearman_df = spearman_report(combined_rsrp_throughput, x_col="rsrp_dbm", y_col="download_mbps", source_col="source")
    write_csv(spearman_df, supplementary_result_dir / "external_spearman_rsrp_throughput.csv")

    if path_exists(vienna_scanner_lte_path) and path_exists(vienna_scanner_5g_path):
        scanner_lte, scanner_5g = read_parquet(vienna_scanner_lte_path), read_parquet(vienna_scanner_5g_path)
        scanner_common = align_vienna_scanner(scanner_lte, scanner_5g)
        scanner_summary = diagnostic_summary({"Vienna_scanner_all": scanner_common, "Vienna_scanner_LTE": scanner_common[scanner_common["network_type"] == "4G"], "Vienna_scanner_5G": scanner_common[scanner_common["network_type"] == "5G"]}, variables=("rsrp_dbm",))
        write_csv(scanner_summary, supplementary_result_dir / "vienna_scanner_rsrp_summary.csv")

    print("Generating external-alignment figures...")

    # LaTeX caption/title candidate: "RSRP ECDF: SynNetQoS vs Vienna Phone Measurements"
    fig_rsrp = plot_ecdf_by_group(
        syn_common,
        vienna_phone_common,
        variable="rsrp_dbm",
        title="RSRP ECDF: SynNetQoS vs Vienna Phone Measurements",
        xlabel="RSRP (dBm)",
        group_pairs=vienna_network_pairs,
        ext_label_prefix="Vienna phone",
        ext_display_names={"4G": "LTE/4G", "5G": "5G serving cell"},
    )
    save_plot(fig_rsrp, figure_dir / "external_rsrp_ecdf_vienna_phone_matched.pdf", dpi=300)

    # LaTeX caption/title candidate: "Download Throughput ECDF: SynNetQoS vs Vienna Phone Measurements"
    fig_vienna_download = plot_ecdf_by_group(
        syn_common,
        vienna_phone_common,
        variable="download_mbps",
        title="Download Throughput ECDF: SynNetQoS vs Vienna Phone Measurements",
        xlabel="Download throughput (Mbps)",
        group_pairs=vienna_network_pairs,
        ext_label_prefix="Vienna phone",
        ext_display_names={"4G": "LTE/4G", "5G": "5G serving cell"},
    )
    save_plot(fig_vienna_download, figure_dir / "external_download_ecdf_vienna_phone_matched.pdf", dpi=300)

    # LaTeX caption/title candidate: "Controlled 5G Download ECDF: SynNetQoS vs Campus QoS"
    fig_campus_download = plot_ecdf(
        syn_campus_common["download_mbps"],
        campus_common["download_mbps"],
        title="Controlled 5G Download ECDF: SynNetQoS vs Campus QoS",
        xlabel="Download throughput (Mbps)",
        syn_label="SynNetQoS controlled 5G subset",
        ext_label="Campus 5G QoS",
    )
    save_plot(fig_campus_download, figure_dir / "external_download_ecdf_syn_vs_campus_controlled.pdf", dpi=300)

    # LaTeX caption/title candidate: "Controlled 5G Jitter ECDF: SynNetQoS vs Campus QoS"
    fig_campus_jitter = plot_ecdf(
        syn_campus_common["jitter_ms"],
        campus_common["jitter_ms"],
        title="Controlled 5G Jitter ECDF: SynNetQoS vs Campus QoS",
        xlabel="Jitter (ms)",
        syn_label="SynNetQoS controlled 5G subset",
        ext_label="Campus 5G QoS",
    )
    save_plot(fig_campus_jitter, figure_dir / "external_jitter_ecdf_syn_vs_campus_controlled.pdf", dpi=300)

    # LaTeX caption/title candidate: "Binned RSRP-to-Throughput Trend"
    fig_trend = plot_binned_median_trend(trend_table, title="Binned RSRP-to-Throughput Trend", xlabel="RSRP (dBm)", ylabel="Median download throughput (Mbps)", source_col="source")
    save_plot(fig_trend, supplementary_figure_dir / "external_rsrp_to_throughput_trend.pdf", dpi=300)

    print("External alignment complete.")
    print(f"Main results saved to: {result_dir}")
    print(f"Main figures saved to: {figure_dir}")
    print(f"Supplementary outputs saved under: {supplementary_result_dir}")

if __name__ == "__main__":
    main()
