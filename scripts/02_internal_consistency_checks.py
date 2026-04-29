from synnetqos.validation import plot_scenario_coverage, plot_mobility_vs_handovers, plot_obstruction_vs_signal, test_vonr_latency
from synnetqos.plotting import setup_plot_style, plot_correlation_heatmap, plot_distributions
from synnetqos.io import path_exists, read_csv, save_plot

def main():
    print("--- Running Internal Consistency Checks ---")
    data_path = "data/synthetic/synnetqos-dataset.csv"
    if not path_exists(data_path):
        print(f"Error: Could not find {data_path}. Run 01_generate_dataset.py first.")
        return

    df = read_csv(data_path)
    supplementary_dir = "figures/supplementary"
    archive_dir = "archive/exploratory_figures"
    setup_plot_style()

    print("Generating primary correlation heatmap...")
    corr_features = [
        "Signal_Strength_dBm", "Link_Capacity_Downlink_Mbps", "Offered_Downlink_Mbps", 
        "Download_Speed_Mbps", "Link_Capacity_Upload_Mbps", "Offered_Upload_Mbps", 
        "Upload_Speed_Mbps", "Latency_ms", "Jitter_ms", "Ping_ms", "Battery_Level_percent", 
        "Temperature_C", "Connected_Duration_min", "Interval_Handover_Count", 
        "Cumulative_Handover_Count", "Data_Usage_MB", "Distance_to_Tower_km"
    ]
    fig_corr = plot_correlation_heatmap(df, corr_features)
    save_plot(fig_corr, f"{supplementary_dir}/correlation_heatmap.pdf")

    print("Generating archived exploratory distribution plots...")
    dist_figs = plot_distributions(df)
    save_plot(dist_figs["latency"], f"{archive_dir}/latency_dist.pdf")
    save_plot(dist_figs["jitter"], f"{archive_dir}/jitter_dist.pdf")

    print("Generating scenario coverage heatmap...")
    fig_coverage = plot_scenario_coverage(df)
    save_plot(fig_coverage, f"{supplementary_dir}/dataset_coverage_heatmap.pdf")
        
    print("Validating physical relationships (Mobility vs Handovers)...")
    fig_mobility = plot_mobility_vs_handovers(df)
    save_plot(fig_mobility, f"{supplementary_dir}/internal_check_mobility_vs_handovers.pdf")
        
    print("Validating physical relationships (Obstruction vs Signal)...")
    fig_obstruction = plot_obstruction_vs_signal(df)
    save_plot(fig_obstruction, f"{supplementary_dir}/internal_check_obstruction_vs_signal.pdf")
        
    print("Running VoNR statistical t-test...")
    fig_vonr, t_stat, p_val = test_vonr_latency(df)
    save_plot(fig_vonr, f"{supplementary_dir}/latency_vonr_boxplot.pdf")

    print(f"--- VoNR t-Test --- \n t-stat = {t_stat:.3f}, p-val = {p_val:.5e}")
    print(f"Internal consistency validation complete. Supplementary outputs saved to '{supplementary_dir}', archived exploratory outputs saved to '{archive_dir}'.")

if __name__ == "__main__":
    main()