import math

import pandas as pd
import numpy as np
import random
from datetime import timedelta
from scipy.special import expit
from synnetqos import config, profiles


def log_distance_path_loss(distance_km: float) -> float:
    # Legacy fallback path-loss model retained for ablation and comparison.
    cfg = config.DEFAULT_CONFIG
    distance_km = max(float(distance_km), cfg.d0_km)
    return cfg.pl0_db + 10.0 * cfg.path_loss_exponent * np.log10(distance_km / cfg.d0_km)


def band_frequency_ghz(band: str) -> float:
    return float(profiles.BAND_FREQUENCY_GHZ.get(band, config.DEFAULT_CONFIG.default_carrier_frequency_ghz))


def band_distance_limit_km(band: str) -> float:
    """Return the maximum serving distance allowed for the selected band.

    The global tower-distance cap remains authoritative. Band-specific limits
    only tighten that cap, which prevents mmWave-like bands from being sampled
    with the same geometry as low-/mid-band deployments.
    """
    cfg = config.DEFAULT_CONFIG
    band_limits = getattr(profiles, "BAND_DISTANCE_LIMITS_KM", {})
    raw_limit = float(band_limits.get(band, cfg.tower_distance_max_km))
    return float(np.clip(raw_limit, 0.05, cfg.tower_distance_max_km))


def sample_initial_tower_distance_km(band: str) -> float:
    cfg = config.DEFAULT_CONFIG
    band_limit = band_distance_limit_km(band)
    sampled_distance = np.random.gamma(
        shape=cfg.tower_distance_gamma_shape,
        scale=cfg.tower_distance_gamma_scale,
    )
    return round(float(np.clip(sampled_distance, 0.05, band_limit)), 2)


def evolve_tower_distance_km(current_distance_km: float, movement_speed: str, band: str) -> float:
    """Evolve tower distance while respecting global and band-specific limits.

    Movement noise is scaled for short-range bands so mmWave sessions do not
    unrealistically spend most timesteps clipped at the upper distance bound.
    """
    band_limit = band_distance_limit_km(band)

    if movement_speed == "Driving":
        sigma = min(0.5, 0.25 * band_limit)
    elif movement_speed == "Walking":
        sigma = min(0.05, 0.08 * band_limit)
    else:
        return float(np.clip(current_distance_km, 0.05, band_limit))

    evolved_distance = current_distance_km + np.random.normal(0.0, sigma)
    return float(np.clip(evolved_distance, 0.05, band_limit))


def distance_pressure(distance_km: float, band: str) -> float:
    """Distance expressed as a fraction of the serving range for the band."""
    return float(np.clip(distance_km / band_distance_limit_km(band), 0.0, 1.5))


def distance_3d_m(distance_2d_m: float, h_bs_m: float, h_ut_m: float) -> float:
    return math.sqrt(distance_2d_m**2 + (h_bs_m - h_ut_m) ** 2)


def tr38901_breakpoint_distance_m(fc_ghz: float, h_bs_m: float, h_ut_m: float) -> float:
    # TR 38.901 uses effective antenna heights h'BS and h'UT above the environmental height.
    # For the UMi default h=1 m, h'UT is clipped to a small positive value because hUT=1.5 m.
    cfg = config.DEFAULT_CONFIG
    environmental_height_m = 1.0
    h_bs_eff = max(h_bs_m - environmental_height_m, 0.1)
    h_ut_eff = max(h_ut_m - environmental_height_m, 0.1)
    return 4.0 * h_bs_eff * h_ut_eff * fc_ghz * 1.0e9 / cfg.light_speed_m_per_s


def tr38901_umi_los_probability(distance_2d_m: float) -> float:
    # UMi-Street Canyon LOS probability used for stochastic LOS/NLOS assignment.
    distance_2d_m = max(float(distance_2d_m), 1.0)
    if distance_2d_m <= 18.0:
        return 1.0
    return float((18.0 / distance_2d_m) + math.exp(-distance_2d_m / 36.0) * (1.0 - 18.0 / distance_2d_m))


def sample_los_state(distance_2d_m: float) -> tuple[str, float]:
    los_probability = tr38901_umi_los_probability(distance_2d_m)
    return ("LOS" if random.random() < los_probability else "NLOS"), los_probability


def tr38901_umi_street_canyon_path_loss(
    distance_km: float,
    carrier_frequency_ghz: float,
    los_state: str,
) -> dict[str, float | str]:
    """Selected 3GPP TR 38.901 UMi-Street Canyon large-scale path-loss mode.

    This implements the large-scale LOS/NLOS path-loss equations, stochastic
    shadow fading, 2D/3D distance handling, and breakpoint-distance logic used
    for UMi-Street Canyon. It does not generate cluster-level small-scale channel
    coefficients or packet-level PHY/MAC traces.
    """
    cfg = config.DEFAULT_CONFIG
    distance_2d_m = float(np.clip(distance_km * 1000.0, cfg.los_min_2d_distance_m, cfg.los_max_2d_distance_m))
    h_bs = cfg.base_station_height_m
    h_ut = cfg.ue_height_m
    fc = float(carrier_frequency_ghz)

    d_3d = distance_3d_m(distance_2d_m, h_bs, h_ut)
    d_bp = tr38901_breakpoint_distance_m(fc, h_bs, h_ut)

    pl_los_1 = 32.4 + 21.0 * math.log10(d_3d) + 20.0 * math.log10(fc)
    pl_los_2 = (
        32.4
        + 40.0 * math.log10(d_3d)
        + 20.0 * math.log10(fc)
        - 9.5 * math.log10(d_bp**2 + (h_bs - h_ut) ** 2)
    )
    pl_los = pl_los_1 if distance_2d_m <= d_bp else pl_los_2

    pl_nlos_candidate = (
        35.3 * math.log10(d_3d)
        + 22.4
        + 21.3 * math.log10(fc)
        - 0.3 * (h_ut - 1.5)
    )

    if los_state == "LOS":
        deterministic_path_loss = pl_los
        shadow_std = cfg.los_shadow_fading_std_db
    else:
        deterministic_path_loss = max(pl_los, pl_nlos_candidate)
        shadow_std = cfg.nlos_shadow_fading_std_db

    shadow_fading_db = float(np.random.normal(0.0, shadow_std))
    path_loss_db = deterministic_path_loss + shadow_fading_db

    return {
        "Propagation_Model": "tr38901_umi_street_canyon",
        "Propagation_Scenario": "UMi_Street_Canyon",
        "LOS_State": los_state,
        "Carrier_Frequency_GHz": fc,
        "Distance_2D_m": distance_2d_m,
        "Distance_3D_m": d_3d,
        "Breakpoint_Distance_m": d_bp,
        "LOS_Probability": tr38901_umi_los_probability(distance_2d_m),
        "Path_Loss_dB": path_loss_db,
        "Deterministic_Path_Loss_dB": deterministic_path_loss,
        "Shadow_Fading_dB": shadow_fading_db,
    }


def radio_path_loss(distance_km: float, band: str) -> dict[str, float | str]:
    cfg = config.DEFAULT_CONFIG
    fc = band_frequency_ghz(band)

    if cfg.propagation_model == "tr38901_umi_street_canyon":
        distance_2d_m = float(np.clip(distance_km * 1000.0, cfg.los_min_2d_distance_m, cfg.los_max_2d_distance_m))
        los_state, los_probability = sample_los_state(distance_2d_m)
        radio = tr38901_umi_street_canyon_path_loss(distance_km, fc, los_state)
        radio["LOS_Probability"] = los_probability
        return radio

    if cfg.propagation_model == "log_distance":
        d2d = max(float(distance_km) * 1000.0, cfg.los_min_2d_distance_m)
        d3d = distance_3d_m(d2d, cfg.base_station_height_m, cfg.ue_height_m)
        path_loss_db = float(log_distance_path_loss(distance_km))
        shadow_fading_db = float(np.random.normal(0.0, cfg.fast_fading_std_db))
        return {
            "Propagation_Model": "log_distance",
            "Propagation_Scenario": "Legacy_Log_Distance",
            "LOS_State": "Not_Applicable",
            "Carrier_Frequency_GHz": fc,
            "Distance_2D_m": d2d,
            "Distance_3D_m": d3d,
            "Breakpoint_Distance_m": np.nan,
            "LOS_Probability": np.nan,
            "Path_Loss_dB": path_loss_db + shadow_fading_db,
            "Deterministic_Path_Loss_dB": path_loss_db,
            "Shadow_Fading_dB": shadow_fading_db,
        }

    raise ValueError(f"Unsupported propagation model: {cfg.propagation_model}")


def contextual_radio_penalties(obstruction: str, weather: str, movement_speed: str, is_indoor: bool) -> dict[str, float]:
    cfg = config.DEFAULT_CONFIG
    obstruction_penalty = {"Low": 2.0, "Medium": 8.0, "High": 18.0}[obstruction]
    weather_penalty = {"Clear": 0.0, "Rainy": 5.0, "Foggy": 8.0}[weather]
    mobility_penalty = {"Static": 0.0, "Walking": 2.0, "Driving": 6.0}[movement_speed]
    indoor_penalty = cfg.indoor_penalty_db if is_indoor else 0.0
    total_penalty = obstruction_penalty + weather_penalty + mobility_penalty + indoor_penalty
    return {
        "Obstruction_Penalty_dB": obstruction_penalty,
        "Weather_Penalty_dB": weather_penalty,
        "Mobility_Penalty_dB": mobility_penalty,
        "Indoor_Penalty_dB": indoor_penalty,
        "Contextual_Penalty_dB": total_penalty,
    }


def generate_radio_link(network_type, distance_km, obstruction, weather, movement_speed, band, is_indoor):
    cfg = config.DEFAULT_CONFIG
    radio = radio_path_loss(distance_km=distance_km, band=band)
    penalties = contextual_radio_penalties(obstruction, weather, movement_speed, is_indoor)

    fast_fading_db = float(np.random.normal(0.0, cfg.fast_fading_std_db))

    # In the TR 38.901 mode, band/frequency effects are already represented through fc.
    # The legacy band-gain shortcut is used only by the legacy log-distance mode.
    band_adjustment_db = profiles.BAND_GAIN_DB.get(band, 0.0) if radio["Propagation_Model"] == "log_distance" else 0.0

    rsrp_unclipped = (
        cfg.effective_tx_power_dbm
        - float(radio["Path_Loss_dB"])
        - penalties["Contextual_Penalty_dB"]
        + band_adjustment_db
        + fast_fading_db
    )
    rsrp_clipped = float(np.clip(rsrp_unclipped, cfg.rsrp_min_dbm, cfg.rsrp_max_dbm))

    radio.update(penalties)
    radio["Effective_TX_Power_dBm"] = cfg.effective_tx_power_dbm
    radio["Band_Adjustment_dB"] = float(band_adjustment_db)
    radio["Fast_Fading_dB"] = fast_fading_db
    radio["Signal_Strength_Unclipped_dBm"] = float(rsrp_unclipped)
    radio["Signal_Strength_dBm"] = rsrp_clipped
    radio["RSRP_Clipped"] = bool(rsrp_unclipped != rsrp_clipped)

    return radio


def generate_latency(signal_strength, congestion, vonr_enabled, network_type, movement_speed):
    base_latency = np.interp(signal_strength, [-120, -70], [300, 15])
    # Scales base latency by network tech, congestion, and mobility multipliers, applying a 15% discount if VoNR is enabled
    factors = {"Low": 1.0, "Medium": 1.5, "High": 2.5}[congestion] * {"5G SA": 0.6, "5G NSA": 1.0, "4G": 1.4}[network_type] * {"Static": 1.0, "Walking": 1.1, "Driving": 1.25}[movement_speed]
    latency = base_latency * factors * (0.85 if vonr_enabled else 1.0)
    return np.clip(latency + np.random.normal(0, 8), 5, 1000)

def generate_jitter(latency, movement_speed, congestion):
    # Computes mean jitter proportional to latency, scales it by mobility and congestion factors, then applies random noise
    mean_jitter = max(0.05, (latency / config.DEFAULT_CONFIG.jitter_latency_divisor) * {"Static": 0.8, "Walking": 1.1, "Driving": 1.6}[movement_speed] * {"Low": 0.8, "Medium": 1.2, "High": 2.0}[congestion])
    return np.clip(np.random.normal(mean_jitter, 0.15 + 0.05 * mean_jitter), 0.02, 100)

def generate_link_capacity(signal_strength, congestion, movement_speed, network_type, latency, ue_profile):
    max_speeds = {"4G": 150, "5G NSA": 700, "5G SA": 900}
    # Estimates capacity by multiplying max theoretical speed by hardware efficiencies and discounting for poor signal, congestion, mobility, and high latency
    capacity = (max_speeds[network_type] * expit((signal_strength + 95) / 5) * {"4G": 0.8, "5G NSA": 1.0, "5G SA": 1.05}[network_type] 
                * profiles.UE_PROFILES.get(ue_profile, profiles.UE_PROFILES["UE-Standard"])["throughput_efficiency"]
                * {"Low": 1.0, "Medium": 0.6, "High": 0.25}[congestion] * {"Static": 1.0, "Walking": 0.9, "Driving": 0.8}[movement_speed] 
                * (expit(-(latency - 150) / 30) * 0.4 + 0.6))
    return max(0, capacity + np.random.normal(0, max_speeds[network_type] * 0.01))

def sample_offered_traffic_mbps(network_type, app_type, direction):
    # Samples user traffic demand from a log-normal distribution using predefined median values for the specific network and app type
    median = profiles.DOWNLINK_DEMAND_MEDIAN_MBPS[network_type][app_type] if direction == "downlink" else profiles.UPLINK_DEMAND_MEDIAN_MBPS[network_type][app_type]
    sample = np.random.lognormal(
        mean=np.log(median),
        sigma=config.DEFAULT_CONFIG.demand_sigma,
    )

    if direction == "downlink":
        return float(np.clip(sample, 0.05, 1500.0))

    return float(np.clip(sample, 0.01, 250.0))

def generate_observed_throughput(link_capacity_mbps, offered_traffic_mbps):
    # Final throughput is bounded by available capacity after stochastic realization.
    base_throughput = min(link_capacity_mbps, offered_traffic_mbps)
    realized_throughput = base_throughput * np.random.uniform(0.90, 1.05)
    return max(0.0, min(realized_throughput, link_capacity_mbps))

def map_video_quality_to_label(score):
    # Maps a heuristic streaming-QoE score (1-5) to descriptive quality labels.
    if pd.isna(score): return np.nan
    return "Excellent" if score >= 4.5 else "Good" if score >= 3.5 else "Fair" if score >= 2.5 else "Poor" if score >= 1.5 else "Bad"

def generate_video_quality(download_speed, latency, jitter, congestion, app_type):
    if app_type != "Streaming": return np.nan
    # Calculates a heuristic streaming-QoE score (1-5). This is not a bitstream-based ITU-T P.1203 implementation.
    score = (0.55 * np.clip(download_speed / 40, 0, 1.0) + 0.25 * np.clip(1 - (latency / 200), 0, 1.0) 
             + 0.15 * np.clip(1 - (jitter / 30), 0, 1.0) + 0.05 * np.random.rand() + {"Low": 0, "Medium": -0.1, "High": -0.3}[congestion])
    return round(np.clip(1 + score * 4, 1, 5), 2)

def generate_dropped_connection(signal_strength, latency, jitter, congestion, network_type, battery, infrastructure_profile, anomaly, tower_load, interval_handovers, distance, band):
    # Uses a logistic risk model (expit) summing weighted penalties for poor signal,
    # latency, infrastructure state, handovers, and band-normalized edge-of-cell pressure.
    edge_of_cell_penalty = 0.8 if distance_pressure(distance, band) > 0.85 else 0.0

    linear_combination = (-20.0 + (0.15 * np.clip(-signal_strength - 85, 0, 50)) + (0.02 * max(latency - 150, 0)) + (0.05 * max(jitter - 25, 0)) 
                          + {"Low": 0, "Medium": 0.8, "High": 1.8}[congestion] + {"4G": 1.0, "5G NSA": 0.5, "5G SA": 0.1}[network_type] 
                          + (2.0 if battery < 15 else 0) + profiles.INFRASTRUCTURE_PROFILES.get(infrastructure_profile, 0) 
                          + (5.0 if anomaly else 0) + {"Light": 0, "Moderate": 0.7, "Heavy": 1.5}[tower_load] 
                          + (0.4 * min(interval_handovers, 8)) + edge_of_cell_penalty + (1.5 if band in ["n257", "n258", "n260"] else 0))
    return random.random() < expit(linear_combination)

def dynamic_congestion(hour, tower_load, movement_speed):
    # Assigns a congestion level based on time of day (peak vs off-peak weights) and forces it higher if tower load or driving speed dictates it
    levels = {"Low": 0, "Medium": 1, "High": 2}
    level = levels[random.choices(["Low", "Medium", "High"], weights=[0.1, 0.5, 0.4] if 8 <= hour <= 11 or 18 <= hour <= 22 else [0.7, 0.25, 0.05] if 0 <= hour <= 6 else [0.4, 0.4, 0.2])[0]]
    if tower_load == "Heavy": level = max(level, levels["High"])
    elif tower_load == "Moderate" or movement_speed == "Driving": level = max(level, levels["Medium"])
    return [k for k, v in levels.items() if v == level][0]

def simulate_network_anomaly(signal_strength, congestion, battery, distance, band):
    # Rare synthetic stress-event flag.
    # The flag is not externally validated and is not used as a primary target label.
    # Its probability increases under weak signal, high congestion, edge-of-cell pressure,
    # and mmWave-like band conditions.
    near_edge = distance_pressure(distance, band) > 0.85
    mmwave_like = band in ["n257", "n258", "n260"]

    stress_score = (
        (1.4 if signal_strength < -115 else 0.0)
        + {"Low": 0.0, "Medium": 0.5, "High": 1.0}[congestion]
        + (1.0 if near_edge else 0.0)
        + (0.6 if mmwave_like else 0.0)
        + (0.5 if battery < 30 else 0.0)
    )

    anomaly_probability = 0.001 + 0.020 * expit(stress_score - 2.3)
    return random.random() < anomaly_probability

def evolve_movement_state(current_speed, transition_matrix):
    # Updates the user's mobility state for the next timestep using the provided Markov chain transition matrix
    transition = transition_matrix.get(current_speed, {"states": ["Static"], "weights": [1.0]})
    return random.choices(transition["states"], weights=transition["weights"], k=1)[0]

def context_aware_handover(movement_speed, tower_load):
    # Simulates interval handover counts using Poisson distributions, with higher expected rates for driving and heavy tower loads
    if movement_speed == "Driving": return np.random.poisson(4 if tower_load == "Heavy" else 2) + 1
    return np.random.poisson(1) if movement_speed == "Walking" else 0

def calculate_energy_drain(data_usage_mb, signal_strength_dbm, network_type, ue_profile):
    # Calculates battery drain percentage per timestep summing idle cost, data cost, poor signal tax, and 5G network tax, scaled by device efficiency
    return (0.02 + (0.001 * data_usage_mb) + (0.08 * np.clip((signal_strength_dbm + 70) / -50, 0, 1)) + (0.03 if "5G" in network_type else 0)) * profiles.UE_PROFILES.get(ue_profile, profiles.UE_PROFILES["UE-Standard"])["energy_efficiency"]

# Master Generation Function
def generate_full_dataset(run_id, num_sessions=5000, session_length=10):
    np.random.seed(run_id)
    random.seed(run_id)
    data = []

    for i in range(num_sessions):
        session_id, user_id = f"S{run_id:03d}_{i:05d}", f"U{run_id:03d}_{i:05d}"
        deployment_area = random.choice(list(profiles.DEPLOYMENT_AREAS.keys()))
        base_lat, base_lon, area_type = profiles.DEPLOYMENT_AREAS[deployment_area]
        ue_profile = random.choice(profiles.UE_PROFILE_NAMES)
        ue_capability_class = profiles.UE_PROFILES[ue_profile]["tier"]
        operator_profile = random.choice(profiles.OPERATOR_AVAILABILITY[deployment_area])
        infrastructure_profile = random.choices(list(profiles.INFRASTRUCTURE_PROFILES.keys()), weights=[0.75, 0.20, 0.05], k=1)[0]
        network_type = random.choices(profiles.NETWORK_TYPES, weights=[0.25, 0.4, 0.35])[0]
        band = random.choice(profiles.OPERATOR_BANDS[operator_profile]) if "5G" in network_type else "LTE_Anchor"
        is_indoor = random.random() < config.DEFAULT_CONFIG.indoor_probability
        session_mobility_matrix = {"Static": {"states": ["Static", "Walking"], "weights": [0.5, 0.5]}, "Walking": {"states": ["Static", "Walking"], "weights": [0.4, 0.6]}} if is_indoor else profiles.MOBILITY_TRANSITION_MATRIX
        
        current_lat, current_lon = base_lat + np.random.normal(0, 0.005), base_lon + np.random.normal(0, 0.005)
        current_movement_speed = random.choices(["Static", "Walking"], weights=[0.6, 0.4])[0]
        current_distance_to_tower = sample_initial_tower_distance_km(band)
        current_obstruction = "High" if is_indoor and random.random() < config.DEFAULT_CONFIG.indoor_high_obstruction_probability else random.choices(["Low", "Medium"], weights=[0.5, 0.5])[0]
        current_battery, session_start = random.uniform(80, 100), config.DEFAULT_CONFIG.start_date + timedelta(days=random.uniform(0, 30), minutes=random.randint(0, 24 * 60))
        session_weather, current_tower_id, cumulative_handover_count = random.choices(profiles.WEATHER_TYPES, weights=[0.7, 0.2, 0.1])[0], f"TWR-{random.randint(1000, 9999)}", 0

        # --- Sub-loop: Generate timestep metrics for the current session ---
        for t in range(session_length):
            timestamp = session_start + timedelta(minutes=t)
            hour, day_of_week = timestamp.hour, timestamp.strftime("%A")
            
            if t > 0 and random.random() < 0.2:
                current_movement_speed = evolve_movement_state(current_movement_speed, session_mobility_matrix)
                if is_indoor and current_movement_speed == "Driving": current_movement_speed = "Walking"

            if current_movement_speed == "Driving":
                current_distance_to_tower = evolve_tower_distance_km(current_distance_to_tower, current_movement_speed, band)
                current_lat = current_lat + np.random.normal(0, 0.003)
            elif current_movement_speed == "Walking":
                current_distance_to_tower = evolve_tower_distance_km(current_distance_to_tower, current_movement_speed, band)
                current_lat = current_lat + np.random.normal(0, 0.0003)
            else:
                current_distance_to_tower = evolve_tower_distance_km(current_distance_to_tower, current_movement_speed, band)

            tower_load, app_type = random.choices(profiles.TOWER_LOAD_LEVELS, weights=[0.5, 0.3, 0.2])[0], random.choice(profiles.APP_TYPES)
            vonr_enabled = (network_type == "5G SA") and (app_type == "Call") and (random.random() > 0.3)
            congestion = dynamic_congestion(hour, tower_load, current_movement_speed)
            interval_handovers = context_aware_handover(current_movement_speed, tower_load)
            cumulative_handover_count += interval_handovers
            if interval_handovers > 0: current_tower_id = f"TWR-{random.randint(1000, 9999)}"

            radio_link = generate_radio_link(network_type, current_distance_to_tower, current_obstruction, session_weather, current_movement_speed, band, is_indoor)
            signal_strength = round(float(radio_link["Signal_Strength_dBm"]), 2)
            anomaly = simulate_network_anomaly(signal_strength, congestion, current_battery, current_distance_to_tower, band)
            latency = round(generate_latency(signal_strength, congestion, vonr_enabled, network_type, current_movement_speed), 2)
            jitter = round(generate_jitter(latency, current_movement_speed, congestion), 2)
            
            link_capacity_downlink = round(generate_link_capacity(signal_strength, congestion, current_movement_speed, network_type, latency, ue_profile), 2)
            offered_downlink = round(sample_offered_traffic_mbps(network_type, app_type, "downlink"), 2)
            download_speed = round(generate_observed_throughput(link_capacity_downlink, offered_downlink), 2)
            link_capacity_upload = round(link_capacity_downlink * profiles.UPLINK_CAPACITY_RATIO[network_type], 2)
            offered_upload = round(sample_offered_traffic_mbps(network_type, app_type, "uplink"), 2)
            upload_speed = round(generate_observed_throughput(link_capacity_upload, offered_upload), 2)
            activity_factor = np.clip(np.random.normal(config.DEFAULT_CONFIG.activity_factor, config.DEFAULT_CONFIG.activity_factor_std), 0.03, 0.18)
            data_usage = round((download_speed * 1 * 60 / 8) * activity_factor, 2)
            
            current_battery = max(5.0, current_battery - calculate_energy_drain(data_usage, signal_strength, network_type, ue_profile))
            temperature = round(np.interp(round(current_battery, 2), [5, 100], [45, 28]) + np.random.normal(0, 1.5), 1)
            video_quality_score = generate_video_quality(download_speed, latency, jitter, congestion, app_type)
            
            dropped = generate_dropped_connection(signal_strength, latency, jitter, congestion, network_type, round(current_battery, 2), infrastructure_profile, anomaly, tower_load, interval_handovers, current_distance_to_tower, band)

            # --- Append the generated timestep row ---
            data.append([
                timestamp, hour, day_of_week, day_of_week in ["Saturday", "Sunday"], user_id, session_id,
                deployment_area, current_lat, current_lon, area_type, signal_strength,
                radio_link["Propagation_Model"], radio_link["Propagation_Scenario"], radio_link["LOS_State"],
                round(float(radio_link["LOS_Probability"]), 4) if not pd.isna(radio_link["LOS_Probability"]) else np.nan,
                round(float(radio_link["Carrier_Frequency_GHz"]), 3),
                round(float(radio_link["Distance_2D_m"]), 2),
                round(float(radio_link["Distance_3D_m"]), 2),
                round(float(radio_link["Breakpoint_Distance_m"]), 2) if not pd.isna(radio_link["Breakpoint_Distance_m"]) else np.nan,
                round(float(radio_link["Path_Loss_dB"]), 2),
                round(float(radio_link["Deterministic_Path_Loss_dB"]), 2),
                round(float(radio_link["Shadow_Fading_dB"]), 2),
                round(float(radio_link["Fast_Fading_dB"]), 2),
                round(float(radio_link["Obstruction_Penalty_dB"]), 2),
                round(float(radio_link["Weather_Penalty_dB"]), 2),
                round(float(radio_link["Mobility_Penalty_dB"]), 2),
                round(float(radio_link["Indoor_Penalty_dB"]), 2),
                round(float(radio_link["Contextual_Penalty_dB"]), 2),
                round(float(radio_link["Effective_TX_Power_dBm"]), 2),
                round(float(radio_link["Signal_Strength_Unclipped_dBm"]), 2),
                bool(radio_link["RSRP_Clipped"]),
                link_capacity_downlink, offered_downlink, download_speed, link_capacity_upload, offered_upload, upload_speed,
                latency, jitter, round(max(0, latency + np.random.normal(0, 4)), 1), network_type, ue_profile,
                ue_capability_class, operator_profile, infrastructure_profile, band, round(current_battery, 2), temperature,
                t + 1, interval_handovers, cumulative_handover_count, round(activity_factor, 3), data_usage, congestion,
                current_movement_speed, app_type, session_weather, current_obstruction, is_indoor, current_tower_id,
                round(float(current_distance_to_tower), 3), tower_load, video_quality_score, map_video_quality_to_label(video_quality_score),
                vonr_enabled, anomaly, dropped
            ])

    cols = [
        "Timestamp", "Hour", "Day_of_Week", "Is_Weekend", "User_ID", "Session_ID", "Deployment_Area",
        "Synthetic_Latitude", "Synthetic_Longitude", "Area_Type", "Signal_Strength_dBm",
        "Propagation_Model", "Propagation_Scenario", "LOS_State", "LOS_Probability", "Carrier_Frequency_GHz",
        "Distance_2D_m", "Distance_3D_m", "Breakpoint_Distance_m", "Path_Loss_dB",
        "Deterministic_Path_Loss_dB", "Shadow_Fading_dB", "Fast_Fading_dB", "Obstruction_Penalty_dB",
        "Weather_Penalty_dB", "Mobility_Penalty_dB", "Indoor_Penalty_dB", "Contextual_Penalty_dB",
        "Effective_TX_Power_dBm", "Signal_Strength_Unclipped_dBm", "RSRP_Clipped",
        "Link_Capacity_Downlink_Mbps", "Offered_Downlink_Mbps", "Download_Speed_Mbps",
        "Link_Capacity_Upload_Mbps", "Offered_Upload_Mbps", "Upload_Speed_Mbps", "Latency_ms", "Jitter_ms",
        "Ping_ms", "Network_Type", "UE_Profile", "UE_Capability_Class", "Operator_Profile", "Infrastructure_Profile",
        "Band", "Battery_Level_percent", "Temperature_C", "Connected_Duration_min", "Interval_Handover_Count",
        "Cumulative_Handover_Count", "Activity_Factor", "Data_Usage_MB", "Congestion_Level", "Movement_Speed",
        "App_Type", "Weather", "Obstruction_Level", "Is_Indoor", "Tower_ID", "Distance_to_Tower_km",
        "Tower_Load", "Video_Quality", "Video_Quality_Label", "VoNR_Enabled", "Anomalous", "Dropped_Connection"
    ]
    return pd.DataFrame(data, columns=cols)
    