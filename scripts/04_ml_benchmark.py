"""Run the final leakage-aware A-D benchmark.

Use the patched final benchmark logic:
- seeds [42, 7, 2026]
- GroupShuffleSplit by Session_ID
- no SMOTE
- dummy threshold fixed at 0.5
- Task C excludes Temperature_C
- Task D inherits Task C exclusions and removes cumulative handovers/offered traffic

Paper-facing outputs go to results/ml_benchmark/.
Optional per-seed prediction files should be disabled by default.
"""

SAVE_SEED_PREDICTIONS = False
SAVE_PER_MODEL_IMPORTANCE = False
SAVE_SUPPLEMENTARY_PLOTS = False


def main() -> None:
    raise NotImplementedError("Move the final patched A-D ML benchmark here.")


if __name__ == "__main__":
    main()
