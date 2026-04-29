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

    pl0_db: float = 61.4
    d0_km: float = 0.01
    path_loss_exponent: float = 3.3
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

# Default instance to be imported across modules
DEFAULT_CONFIG = GeneratorConfig()
