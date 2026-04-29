from synnetqos.config import GeneratorConfig
from synnetqos.generator import generate_full_dataset
from synnetqos.audits import dataset_integrity_summary, dataset_schema, numerical_range_summary, outlier_report
from synnetqos.io import write_csv, save_json, sha256_of_file

def main() -> None:
    print("Generating primary synthetic dataset...")
    config = GeneratorConfig()
    
    # Generate the dataset
    df = generate_full_dataset(
        run_id=config.seed, 
        num_sessions=config.num_sessions, 
        session_length=config.session_length
    )

    print("Executing silent integrity audits and writing to disk...")
    
    analysis_features = [
        "Signal_Strength_dBm", "Link_Capacity_Downlink_Mbps", "Offered_Downlink_Mbps", 
        "Download_Speed_Mbps", "Link_Capacity_Upload_Mbps", "Offered_Upload_Mbps", 
        "Upload_Speed_Mbps", "Latency_ms", "Jitter_ms", "Ping_ms", "Battery_Level_percent", 
        "Temperature_C", "Connected_Duration_min", "Interval_Handover_Count", 
        "Cumulative_Handover_Count", "Data_Usage_MB", "Distance_to_Tower_km"
    ]

    # Clean I/O execution
    data_path = "data/synthetic/synnetqos-dataset.csv"
    write_csv(df, data_path)
    write_csv(dataset_integrity_summary(df, config.num_sessions, config.session_length), "results/generator/dataset_integrity_summary.csv")
    write_csv(dataset_schema(df), "results/generator/dataset_schema.csv")
    write_csv(numerical_range_summary(df), "results/generator/numerical_range_summary.csv")
    write_csv(outlier_report(df, analysis_features), "results/generator/outlier_report.csv")
    
    # Calculate dataset cryptographic hash
    dataset_hash = sha256_of_file(data_path)
    print(f"Dataset SHA-256 Hash: {dataset_hash}")
    
    # Save the exact configuration + hash for reproducibility
    config_dict = config.__dict__.copy()
    config_dict["start_date"] = str(config.start_date)
    config_dict["dataset_sha256"] = dataset_hash
    save_json(config_dict, "results/generator/generator_config.json")
    
    print("Dataset, generator configuration, and audit summaries successfully saved.")

if __name__ == "__main__":
    main()
    