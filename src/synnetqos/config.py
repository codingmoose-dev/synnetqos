from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    num_sessions: int = 5000
    session_length: int = 10
    start_date: datetime = datetime(2025, 7, 26)
    activity_factor: float = 0.10
    activity_factor_std: float = 0.025

    # Propagation model selection.
    # "tr38901_umi_street_canyon" is the default release model.
    # "log_distance" is retained only for ablation and backward-comparison runs.
    propagation_model: str = "tr38901_umi_street_canyon"

    # Legacy log-distance settings retained for ablation/comparison.
    pl0_db: float = 61.4
    d0_km: float = 0.01
    path_loss_exponent: float = 3.3

    # 3GPP TR 38.901 UMi-Street Canyon large-scale propagation settings.
    # Distances in the TR path-loss equations are in meters and carrier frequency is in GHz.
    effective_tx_power_dbm: float = 44.0
    base_station_height_m: float = 10.0
    ue_height_m: float = 1.5
    light_speed_m_per_s: float = 3.0e8
    default_carrier_frequency_ghz: float = 3.5
    los_shadow_fading_std_db: float = 4.0
    nlos_shadow_fading_std_db: float = 7.82
    fast_fading_std_db: float = 2.0
    los_min_2d_distance_m: float = 10.0
    los_max_2d_distance_m: float = 5000.0

    # Effective RSRP-like clipping range for the released KPI-level dataset.
    rsrp_min_dbm: float = -135.0
    rsrp_max_dbm: float = -50.0

    indoor_probability: float = 0.35
    indoor_penalty_db: float = 10.0
    indoor_high_obstruction_probability: float = 0.45

    tower_distance_gamma_shape: float = 2.0
    tower_distance_gamma_scale: float = 0.35
    tower_distance_max_km: float = 2.0

    jitter_latency_divisor: float = 450.0
    demand_sigma: float = 0.9


# Default instance to be imported across modules.
DEFAULT_CONFIG = GeneratorConfig()
