from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DEFAULT_SEEDS: tuple[int, ...] = (42, 52, 62)

GROUP_COLUMN = "Session_ID"
FUTURE_TARGET_COLUMN = "Drop_t_plus_1"

IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "User_ID",
    "Session_ID",
    "Timestamp",
    "Synthetic_Latitude",
    "Synthetic_Longitude",
    "Tower_ID",
)

TARGET_COLUMNS: tuple[str, ...] = (
    "Dropped_Connection",
    FUTURE_TARGET_COLUMN,
)

LABEL_OR_AUDIT_COLUMNS: tuple[str, ...] = (
    "Video_Quality_Label",
    "Anomalous",
)

GENERATOR_INTERNAL_OR_DEMAND_COLUMNS: tuple[str, ...] = (
    "Offered_Downlink_Mbps",
    "Offered_Upload_Mbps",
    "Propagation_Model",
    "Propagation_Scenario",
    "LOS_State",
    "LOS_Probability",
    "Carrier_Frequency_GHz",
    "Distance_2D_m",
    "Distance_3D_m",
    "Breakpoint_Distance_m",
    "Path_Loss_dB",
    "Deterministic_Path_Loss_dB",
    "Shadow_Fading_dB",
    "Fast_Fading_dB",
    "Obstruction_Penalty_dB",
    "Weather_Penalty_dB",
    "Mobility_Penalty_dB",
    "Indoor_Penalty_dB",
    "Contextual_Penalty_dB",
    "Effective_TX_Power_dBm",
    "Signal_Strength_Unclipped_dBm",
    "RSRP_Clipped",
)

CONTEXT_ONLY_FEATURES: tuple[str, ...] = (
    "Hour",
    "Day_of_Week",
    "Is_Weekend",
    "Deployment_Area",
    "Area_Type",
    "Network_Type",
    "UE_Profile",
    "UE_Capability_Class",
    "Operator_Profile",
    "Weather",
    "Movement_Speed",
    "App_Type",
    "Is_Indoor",
    "Activity_Factor",
    "Connected_Duration_min",
)

METRIC_COLUMNS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "mcc",
    "brier_score",
)

@dataclass(frozen=True)
class MLTask:
    task_id: str
    task_label: str
    target: str
    feature_set: str
    description: str
    interpretation: str
    exclude_columns: tuple[str, ...] = ()
    include_columns: tuple[str, ...] | None = None


def require_columns(df: pd.DataFrame, required: Sequence[str], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{dataset_name} is missing required columns: {missing}")


"""
Prepare same-timestep and one-step-ahead dropped-connection targets.
This function creates Drop_t_plus_1 but does not remove the final row of each session. Task-specific selection removes rows only when the selected target is missing.
"""
def prepare_drop_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = [GROUP_COLUMN, "Connected_Duration_min", "Dropped_Connection"]
    require_columns(df, required, "SynNetQoS ML benchmark")

    out = df.copy()

    if "Band" in out.columns:
        out["Band"] = out["Band"].fillna("Unknown")

    bool_cols = out.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        out[col] = out[col].fillna(False).astype(int)

    out = out.sort_values([GROUP_COLUMN, "Connected_Duration_min"]).copy()
    out[FUTURE_TARGET_COLUMN] = out.groupby(GROUP_COLUMN)["Dropped_Connection"].shift(-1)

    return out


def default_drop_tasks(df: pd.DataFrame) -> list[MLTask]:
    common_exclusions = tuple(
        col
        for col in (
            *IDENTIFIER_COLUMNS,
            *TARGET_COLUMNS,
            *LABEL_OR_AUDIT_COLUMNS,
            *GENERATOR_INTERNAL_OR_DEMAND_COLUMNS,
        )
        if col in df.columns
    )

    context_features = tuple(col for col in CONTEXT_ONLY_FEATURES if col in df.columns)

    return [
        MLTask(
            task_id="A_current_drop_full_qos",
            task_label="A",
            target="Dropped_Connection",
            feature_set="full_current_qos",
            description=(
                "Same-timestep dropped-connection label recovery from full current "
                "QoS/QoE variables. This is an upper-bound internal-consistency task, "
                "not a deployment prediction task."
            ),
            interpretation="upper_bound_internal_consistency",
            exclude_columns=common_exclusions,
        ),
        MLTask(
            task_id="B_future_drop_full_current_qos",
            task_label="B",
            target=FUTURE_TARGET_COLUMN,
            feature_set="full_current_qos_without_current_drop_label",
            description=(
                "One-step-ahead dropped-connection prediction using current-row "
                "QoS/QoE variables, while excluding the current and future drop labels."
            ),
            interpretation="full_information_future_prediction",
            exclude_columns=common_exclusions,
        ),
        MLTask(
            task_id="C_future_drop_context_only",
            task_label="C",
            target=FUTURE_TARGET_COLUMN,
            feature_set="context_only_pre_qos",
            description=(
                "Leakage-controlled one-step-ahead dropped-connection prediction using "
                "only session, scenario, device, application, and temporal context "
                "variables available before QoS outcomes."
            ),
            interpretation="leakage_controlled_future_prediction",
            include_columns=context_features,
        ),
    ]


def task_definitions_table(tasks: Sequence[MLTask]) -> pd.DataFrame:
    rows = []

    for task in tasks:
        row = asdict(task)
        row["exclude_columns"] = "; ".join(task.exclude_columns)
        row["include_columns"] = "" if task.include_columns is None else "; ".join(task.include_columns)
        rows.append(row)

    return pd.DataFrame(rows)


def select_task_data(
    df: pd.DataFrame,
    task: MLTask,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    require_columns(df, [GROUP_COLUMN, task.target], f"task {task.task_id}")

    if task.include_columns is not None:
        feature_columns = [col for col in task.include_columns if col in df.columns]
    else:
        excluded = set(task.exclude_columns)
        feature_columns = [col for col in df.columns if col not in excluded]

    if not feature_columns:
        raise ValueError(f"No usable features remain for task {task.task_id}.")

    required_columns = feature_columns + [task.target, GROUP_COLUMN]
    task_df = df.loc[:, required_columns].dropna(subset=[task.target]).copy()

    X = task_df.loc[:, feature_columns].copy()
    y = task_df[task.target].astype(int)
    groups = task_df[GROUP_COLUMN].astype(str)

    return X, y, groups, feature_columns


def make_group_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int = RANDOM_STATE,
    test_size: float = 0.20,
    validation_size_of_trainval: float = 0.25,
) -> dict[str, pd.Index]:
    # Return session-wise 60/20/20 train/validation/test indices.
    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    trainval_pos, test_pos = next(splitter_1.split(X, y, groups))

    X_trainval = X.iloc[trainval_pos]
    y_trainval = y.iloc[trainval_pos]
    groups_trainval = groups.iloc[trainval_pos]

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size_of_trainval,
        random_state=seed,
    )
    train_pos_rel, val_pos_rel = next(splitter_2.split(X_trainval, y_trainval, groups_trainval))

    train_index = X_trainval.iloc[train_pos_rel].index
    val_index = X_trainval.iloc[val_pos_rel].index
    test_index = X.iloc[test_pos].index

    _assert_no_group_overlap(
        groups.loc[train_index],
        groups.loc[val_index],
        groups.loc[test_index],
    )

    return {
        "train": train_index,
        "validation": val_index,
        "test": test_index,
    }


def _assert_no_group_overlap(
    train_groups: pd.Series,
    val_groups: pd.Series,
    test_groups: pd.Series,
) -> None:
    train_set = set(train_groups)
    val_set = set(val_groups)
    test_set = set(test_groups)

    overlaps = {
        "train_validation_overlap": len(train_set & val_set),
        "train_test_overlap": len(train_set & test_set),
        "validation_test_overlap": len(val_set & test_set),
    }

    if any(overlaps.values()):
        raise RuntimeError(f"Group leakage detected: {overlaps}")


# Replace non-finite numeric values before imputation/scaling.
def sanitize_numeric_values(X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.replace([np.inf, -np.inf], np.nan)

    return np.where(np.isfinite(X), X, np.nan)


def make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X_train.columns if col not in numeric_features]

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        (
                            "sanitize",
                            FunctionTransformer(
                                sanitize_numeric_values,
                                validate=False,
                                feature_names_out="one-to-one",
                            ),
                        ),
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

def build_model_specs(seed: int = RANDOM_STATE) -> dict[str, object]:
    return {
        "Dummy stratified": DummyClassifier(
            strategy="stratified",
            random_state=seed,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=120,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=seed,
        ),
    }

def positive_class_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    # Return positive-class probabilities from probability-capable classifiers.
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{model.__class__.__name__} does not expose predict_proba(). "
            "Use probability-capable classifiers in this benchmark."
        )

    prob = model.predict_proba(X)[:, 1]
    return np.clip(prob, 1e-9, 1 - 1e-9)

def safe_roc_auc(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(roc_auc_score(y_true, y_prob))


def safe_average_precision(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(average_precision_score(y_true, y_prob))


def find_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.5, np.nan

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5, np.nan

    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1_scores[:-1]))

    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate_binary_predictions(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "average_precision": safe_average_precision(y_true, y_prob),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_leakage_audit(df: pd.DataFrame, tasks: Sequence[MLTask]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    for task in tasks:
        if task.include_columns is not None:
            included = set(col for col in task.include_columns if col in df.columns)
        else:
            included = set(df.columns) - set(task.exclude_columns)

        for column in df.columns:
            if column == task.target:
                status = "target"
                reason = "Prediction target for this task."
            elif column == GROUP_COLUMN:
                status = "group_id"
                reason = "Used only for session-wise splitting; never used as a model feature."
            elif column in TARGET_COLUMNS:
                status = "excluded"
                reason = "Current/future target label excluded to prevent direct label leakage."
            elif column in IDENTIFIER_COLUMNS:
                status = "excluded"
                reason = "Identifier or location-like field excluded to reduce memorization/spurious leakage."
            elif column in LABEL_OR_AUDIT_COLUMNS:
                status = "excluded"
                reason = "Label-like or generator-audit field excluded from publication-facing ML features."
            elif task.include_columns is not None and column not in included:
                status = "excluded"
                reason = "Excluded from the conservative context-only feature set."
            elif column in included:
                status = "included"
                reason = "Included in the defined task feature set."
            else:
                status = "excluded"
                reason = "Excluded by task configuration."

            rows.append(
                {
                    "task_id": task.task_id,
                    "task_label": task.task_label,
                    "target": task.target,
                    "feature_set": task.feature_set,
                    "column": column,
                    "status": status,
                    "reason": reason,
                }
            )

    return pd.DataFrame(rows)


def build_feature_set_table(df: pd.DataFrame, tasks: Sequence[MLTask]) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []

    for task in tasks:
        X, _, _, feature_columns = select_task_data(df, task)

        for column in feature_columns:
            rows.append(
                {
                    "task_id": task.task_id,
                    "task_label": task.task_label,
                    "target": task.target,
                    "feature_set": task.feature_set,
                    "feature": column,
                    "dtype": str(X[column].dtype),
                }
            )

    return pd.DataFrame(rows)


def extract_feature_importance(
    model: Pipeline,
    task: MLTask,
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    fitted_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocess"]

    if hasattr(fitted_model, "feature_importances_"):
        values = fitted_model.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(fitted_model, "coef_"):
        values = np.abs(fitted_model.coef_[0])
        importance_type = "absolute_coefficient"
    else:
        return pd.DataFrame(
            columns=[
                "task_id",
                "task_label",
                "model",
                "seed",
                "feature",
                "importance",
                "importance_type",
                "rank",
            ]
        )

    feature_names = preprocessor.get_feature_names_out()

    out = pd.DataFrame(
        {
            "task_id": task.task_id,
            "task_label": task.task_label,
            "model": model_name,
            "seed": seed,
            "feature": feature_names,
            "importance": values,
            "importance_type": importance_type,
        }
    ).sort_values("importance", ascending=False)

    out["rank"] = np.arange(1, len(out) + 1)

    return out


def run_ml_benchmark(
    raw_df: pd.DataFrame,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    model_names: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame | list[dict[str, object]]]:
    df = prepare_drop_prediction_frame(raw_df)
    tasks = default_drop_tasks(df)

    selected_models = set(model_names) if model_names is not None else None

    run_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    importance_frames: list[pd.DataFrame] = []

    for task in tasks:
        X, y, groups, feature_columns = select_task_data(df, task)

        for seed in seeds:
            split = make_group_train_val_test_split(X, y, groups, seed=seed)

            X_train = X.loc[split["train"]]
            X_val = X.loc[split["validation"]]
            X_test = X.loc[split["test"]]

            y_train = y.loc[split["train"]]
            y_val = y.loc[split["validation"]]
            y_test = y.loc[split["test"]]

            train_groups = groups.loc[split["train"]]
            val_groups = groups.loc[split["validation"]]
            test_groups = groups.loc[split["test"]]

            split_rows.append(
                {
                    "task_id": task.task_id,
                    "task_label": task.task_label,
                    "seed": seed,
                    "split_strategy": "GroupShuffleSplit by Session_ID; 60/20/20 train/validation/test",
                    "n_train": int(len(X_train)),
                    "n_validation": int(len(X_val)),
                    "n_test": int(len(X_test)),
                    "n_train_groups": int(train_groups.nunique()),
                    "n_validation_groups": int(val_groups.nunique()),
                    "n_test_groups": int(test_groups.nunique()),
                    "train_positive_rate": float(y_train.mean()),
                    "validation_positive_rate": float(y_val.mean()),
                    "test_positive_rate": float(y_test.mean()),
                    "n_raw_features": int(len(feature_columns)),
                }
            )

            models = build_model_specs(seed=seed)

            if selected_models is not None:
                models = {
                    name: model
                    for name, model in models.items()
                    if name in selected_models
                }

            for model_name, estimator in models.items():
                pipeline = Pipeline(
                    steps=[
                        ("preprocess", make_preprocessor(X_train)),
                        ("model", estimator),
                    ]
                )

                pipeline.fit(X_train, y_train)

                val_prob = positive_class_probability(pipeline, X_val)
                test_prob = positive_class_probability(pipeline, X_test)

                threshold, validation_best_f1 = find_best_threshold(y_val, val_prob)
                metrics = evaluate_binary_predictions(y_test, test_prob, threshold)

                run_row = {
                    "task_id": task.task_id,
                    "task_label": task.task_label,
                    "target": task.target,
                    "feature_set": task.feature_set,
                    "interpretation": task.interpretation,
                    "model": model_name,
                    "seed": seed,
                    "split_strategy": "group_session_holdout_60_20_20",
                    "validation_best_f1": validation_best_f1,
                    "n_train": int(len(X_train)),
                    "n_validation": int(len(X_val)),
                    "n_test": int(len(X_test)),
                    "n_raw_features": int(len(feature_columns)),
                    "n_encoded_features": int(
                        len(pipeline.named_steps["preprocess"].get_feature_names_out())
                    ),
                    "train_positive_rate": float(y_train.mean()),
                    "validation_positive_rate": float(y_val.mean()),
                    "test_positive_rate": float(y_test.mean()),
                }
                run_row.update(metrics)
                run_rows.append(run_row)

                confusion_rows.append(
                    {
                        "task_id": task.task_id,
                        "task_label": task.task_label,
                        "model": model_name,
                        "seed": seed,
                        "threshold": metrics["threshold"],
                        "tn": metrics["tn"],
                        "fp": metrics["fp"],
                        "fn": metrics["fn"],
                        "tp": metrics["tp"],
                    }
                )

                if seed == seeds[0]:
                    curve_rows.append(
                        {
                            "task_id": task.task_id,
                            "task_label": task.task_label,
                            "model": model_name,
                            "seed": seed,
                            "y_true": y_test.to_numpy(),
                            "y_prob": test_prob,
                        }
                    )

                importance = extract_feature_importance(
                    model=pipeline,
                    task=task,
                    model_name=model_name,
                    seed=seed,
                )

                if not importance.empty:
                    importance_frames.append(importance)

    run_metrics = pd.DataFrame(run_rows)
    summary = summarize_run_metrics(run_metrics)

    if importance_frames:
        feature_importance = pd.concat(importance_frames, ignore_index=True)
    else:
        feature_importance = pd.DataFrame()

    return {
        "prepared_df": df,
        "task_definitions": task_definitions_table(tasks),
        "feature_sets": build_feature_set_table(df, tasks),
        "leakage_audit": build_leakage_audit(df, tasks),
        "split_summary": pd.DataFrame(split_rows),
        "run_metrics": run_metrics,
        "summary": summary,
        "confusion_summary": pd.DataFrame(confusion_rows),
        "feature_importance": feature_importance,
        "curve_rows": curve_rows,
    }


def summarize_run_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    if run_metrics.empty:
        return pd.DataFrame()

    group_cols = [
        "task_id",
        "task_label",
        "target",
        "feature_set",
        "interpretation",
        "model",
        "split_strategy",
    ]

    aggregate_cols = [col for col in METRIC_COLUMNS if col in run_metrics.columns]

    summary = (
        run_metrics.groupby(group_cols, dropna=False)[aggregate_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    counts = (
        run_metrics.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("seed", "nunique"),
            n_test_mean=("n_test", "mean"),
            test_positive_rate_mean=("test_positive_rate", "mean"),
            n_raw_features_mean=("n_raw_features", "mean"),
            n_encoded_features_mean=("n_encoded_features", "mean"),
        )
        .reset_index()
    )

    out = counts.merge(summary, on=group_cols, how="left")

    sort_cols = ["task_label", "average_precision_mean", "f1_mean"]
    available_sort_cols = [col for col in sort_cols if col in out.columns]

    return out.sort_values(
        available_sort_cols,
        ascending=[True, False, False][: len(available_sort_cols)],
    ).reset_index(drop=True)