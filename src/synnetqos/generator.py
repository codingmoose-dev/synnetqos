import pandas as pd
import numpy as np
import random
from datetime import timedelta
from scipy.special import expit
from synnetqos import config, profiles

def log_distance_path_loss(distance_km):
    # Calculates path loss using the standard log-distance formula: PL = PL0 + 10 * gamma * log10(d/d0)
    return config.DEFAULT_CONFIG.pl0_db + 10 * config.DEFAULT_CONFIG.path_loss_exponent * np.log10(max(distance_km, config.DEFAULT_CONFIG.d0_km) / config.DEFAULT_CONFIG.d0_km)

def generate_signal_strength(network_type, distance_km, obstruction, weather, movement_speed, band, is_indoor):
    tx_power = {"5G SA": 43, "5G NSA": 43, "4G": 43}
    obstruction_penalty = {"Low": 2, "Medium": 8, "High": 18}[obstruction]
    weather_penalty = {"Clear": 0, "Rainy": 5, "Foggy": 8}[weather]
    mobility_penalty = {"Static": 0, "Walking": 2, "Driving": 6}[movement_speed]
    
    # RSRP = Base TX Power - Path Loss - Contextual Penalties (weather, mobility, indoor) + Band Gain + Random Fading Noise
    rsrp = (tx_power[network_type] - log_distance_path_loss(distance_km) - obstruction_penalty - weather_penalty 
            - mobility_penalty + profiles.BAND_GAIN_DB.get(band, 0) - (config.DEFAULT_CONFIG.indoor_penalty_db if is_indoor else 0) + np.random.normal(0, 4))
    return np.clip(rsrp, config.DEFAULT_CONFIG.rsrp_min_dbm, config.DEFAULT_CONFIG.rsrp_max_dbm)

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
    return np.random.lognormal(mean=np.log(median), sigma=config.DEFAULT_CONFIG.demand_sigma)

def generate_observed_throughput(link_capacity_mbps, offered_traffic_mbps):
    # Final throughput is the minimum of available capacity and user demand, with a slight +/- random fluctuation (90% to 105%)
    return max(0, min(link_capacity_mbps, offered_traffic_mbps) * np.random.uniform(0.90, 1.05))

def map_video_quality_to_label(score):
    # Maps a numerical MOS (Mean Opinion Score, 1-5) to standard descriptive quality labels
    if pd.isna(score): return np.nan
    return "Excellent" if score >= 4.5 else "Good" if score >= 3.5 else "Fair" if score >= 2.5 else "Poor" if score >= 1.5 else "Bad"

def generate_video_quality(download_speed, latency, jitter, congestion, app_type):
    if app_type != "Streaming": return np.nan
    # Calculates a synthetic MOS score (1-5) using weighted normalized contributions from speed, latency, jitter, and congestion penalties
    score = (0.55 * np.clip(download_speed / 40, 0, 1.0) + 0.25 * np.clip(1 - (latency / 200), 0, 1.0) 
             + 0.15 * np.clip(1 - (jitter / 30), 0, 1.0) + 0.05 * np.random.rand() + {"Low": 0, "Medium": -0.1, "High": -0.3}[congestion])
    return round(np.clip(1 + score * 4, 1, 5), 2)

def generate_dropped_connection(signal_strength, latency, jitter, congestion, network_type, battery, infrastructure_profile, anomaly, tower_load, interval_handovers, distance, band):
    # Uses a logistic risk model (expit) summing weighted penalties for poor signal, latency, hardware states, and handovers to compute drop probability
    linear_combination = (-20.0 + (0.15 * np.clip(-signal_strength - 85, 0, 50)) + (0.02 * max(latency - 150, 0)) + (0.05 * max(jitter - 25, 0)) 
                          + {"Low": 0, "Medium": 0.8, "High": 1.8}[congestion] + {"4G": 1.0, "5G NSA": 0.5, "5G SA": 0.1}[network_type] 
                          + (2.0 if battery < 15 else 0) + profiles.INFRASTRUCTURE_PROFILES.get(infrastructure_profile, 0) 
                          + (5.0 if anomaly else 0) + {"Light": 0, "Moderate": 0.7, "Heavy": 1.5}[tower_load] 
                          + (0.4 * min(interval_handovers, 8)) + (0.8 if distance > 4.0 else 0) + (1.5 if band in ["n257", "n258", "n260"] else 0))
    return random.random() < expit(linear_combination)

def dynamic_congestion(hour, tower_load, movement_speed):
    # Assigns a congestion level based on time of day (peak vs off-peak weights) and forces it higher if tower load or driving speed dictates it
    levels = {"Low": 0, "Medium": 1, "High": 2}
    level = levels[random.choices(["Low", "Medium", "High"], weights=[0.1, 0.5, 0.4] if 8 <= hour <= 11 or 18 <= hour <= 22 else [0.7, 0.25, 0.05] if 0 <= hour <= 6 else [0.4, 0.4, 0.2])[0]]
    if tower_load == "Heavy": level = max(level, levels["High"])
    elif tower_load == "Moderate" or movement_speed == "Driving": level = max(level, levels["Medium"])
    return [k for k, v in levels.items() if v == level][0]

def simulate_network_anomaly(signal_strength, congestion, battery, distance):
    # Triggers a 40% chance of an anomaly ONLY under combined extreme conditions (weak signal, low battery, high congestion, long distance)
    return random.random() < 0.4 if signal_strength < -115 and battery < 20 and congestion == "High" and distance > 3.5 else False

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
        band = random.choice(profiles.OPERATOR_BANDS[operator_profile]) if "5G" in network_type else "N/A"
        is_indoor = random.random() < config.DEFAULT_CONFIG.indoor_probability
        session_mobility_matrix = {"Static": {"states": ["Static", "Walking"], "weights": [0.5, 0.5]}, "Walking": {"states": ["Static", "Walking"], "weights": [0.4, 0.6]}} if is_indoor else profiles.MOBILITY_TRANSITION_MATRIX
        
        current_lat, current_lon = base_lat + np.random.normal(0, 0.005), base_lon + np.random.normal(0, 0.005)
        current_movement_speed = random.choices(["Static", "Walking"], weights=[0.6, 0.4])[0]
        current_distance_to_tower = round(float(np.clip(np.random.gamma(shape=config.DEFAULT_CONFIG.tower_distance_gamma_shape, scale=config.DEFAULT_CONFIG.tower_distance_gamma_scale), 0.05, config.DEFAULT_CONFIG.tower_distance_max_km)), 2)
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
                current_distance_to_tower, current_lat = max(0.05, current_distance_to_tower + np.random.normal(0, 0.5)), current_lat + np.random.normal(0, 0.003)
            elif current_movement_speed == "Walking":
                current_distance_to_tower, current_lat = max(0.05, current_distance_to_tower + np.random.normal(0, 0.05)), current_lat + np.random.normal(0, 0.0003)

            tower_load, app_type = random.choices(profiles.TOWER_LOAD_LEVELS, weights=[0.5, 0.3, 0.2])[0], random.choice(profiles.APP_TYPES)
            vonr_enabled = (network_type == "5G SA") and (app_type == "Call") and (random.random() > 0.3)
            congestion = dynamic_congestion(hour, tower_load, current_movement_speed)
            interval_handovers = context_aware_handover(current_movement_speed, tower_load)
            cumulative_handover_count += interval_handovers
            if interval_handovers > 0: current_tower_id = f"TWR-{random.randint(1000, 9999)}"

            signal_strength = round(generate_signal_strength(network_type, current_distance_to_tower, current_obstruction, session_weather, current_movement_speed, band, is_indoor), 2)
            anomaly = simulate_network_anomaly(signal_strength, congestion, current_battery, current_distance_to_tower)
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
            data.append([timestamp, hour, day_of_week, day_of_week in ["Saturday", "Sunday"], user_id, session_id, deployment_area, current_lat, current_lon, area_type, signal_strength, link_capacity_downlink, offered_downlink, download_speed, link_capacity_upload, offered_upload, upload_speed, latency, jitter, round(max(0, latency + np.random.normal(0, 4)), 1), network_type, ue_profile, ue_capability_class, operator_profile, infrastructure_profile, band, round(current_battery, 2), temperature, t + 1, interval_handovers, cumulative_handover_count, round(activity_factor, 3), data_usage, congestion, current_movement_speed, app_type, session_weather, current_obstruction, is_indoor, current_tower_id, current_distance_to_tower, tower_load, video_quality_score, map_video_quality_to_label(video_quality_score), vonr_enabled, anomaly, dropped])

    cols = ["Timestamp", "Hour", "Day_of_Week", "Is_Weekend", "User_ID", "Session_ID", "Deployment_Area", "Synthetic_Latitude", "Synthetic_Longitude", "Area_Type", "Signal_Strength_dBm", "Link_Capacity_Downlink_Mbps", "Offered_Downlink_Mbps", "Download_Speed_Mbps", "Link_Capacity_Upload_Mbps", "Offered_Upload_Mbps", "Upload_Speed_Mbps", "Latency_ms", "Jitter_ms", "Ping_ms", "Network_Type", "UE_Profile", "UE_Capability_Class", "Operator_Profile", "Infrastructure_Profile", "Band", "Battery_Level_percent", "Temperature_C", "Connected_Duration_min", "Interval_Handover_Count", "Cumulative_Handover_Count", "Activity_Factor", "Data_Usage_MB", "Congestion_Level", "Movement_Speed", "App_Type", "Weather", "Obstruction_Level", "Is_Indoor", "Tower_ID", "Distance_to_Tower_km", "Tower_Load", "Video_Quality", "Video_Quality_Label", "VoNR_Enabled", "Anomalous", "Dropped_Connection"]
    return pd.DataFrame(data, columns=cols)
    