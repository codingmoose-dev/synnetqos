from synnetqos.audits import dataset_integrity_summary, dataset_schema, numerical_range_summary
from synnetqos.config import GeneratorConfig
from synnetqos.generator import generate_full_dataset
from synnetqos.io import save_json, write_csv


def main() -> None:
    config = GeneratorConfig()
    df = generate_full_dataset(config)

    write_csv(df, "data/synthetic/synnetqos-dataset.csv")
    write_csv(dataset_integrity_summary(df, config.num_sessions, config.session_length),
              "results/generator/dataset_integrity_summary.csv")
    write_csv(numerical_range_summary(df), "results/generator/numerical_range_summary.csv")
    write_csv(dataset_schema(df), "results/generator/dataset_schema.csv")
    save_json(config.__dict__ | {"start_date": str(config.start_date)},
              "results/generator/generator_config.json")


if __name__ == "__main__":
    main()
