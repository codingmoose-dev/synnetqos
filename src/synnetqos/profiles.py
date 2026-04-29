DEPLOYMENT_AREAS = {"Area_A": (19.0760, 72.8777, "Urban"), "Area_B": (28.6139, 77.2090, "Urban"), "Area_C": (40.7128, -74.0060, "Urban"), "Area_D": (35.6895, 139.6917, "Urban"), "Area_E": (52.5200, 13.4050, "Urban"), "Area_F": (37.7749, -122.4194, "Urban")}
NETWORK_TYPES = ["4G", "5G NSA", "5G SA"]
WEATHER_TYPES = ["Clear", "Rainy", "Foggy"]
OBSTRUCTION_LEVELS = ["Low", "Medium", "High"]
MOVEMENT_SPEEDS = ["Static", "Walking", "Driving"]
APP_TYPES = ["Streaming", "Browse", "Gaming", "Call"]
TOWER_LOAD_LEVELS = ["Light", "Moderate", "Heavy"]

OPERATOR_AVAILABILITY = {"Area_A": ["Operator_A", "Operator_B", "Operator_C"], "Area_B": ["Operator_A", "Operator_B", "Operator_D"], "Area_C": ["Operator_E", "Operator_F", "Operator_G"], "Area_D": ["Operator_H", "Operator_I", "Operator_J"], "Area_E": ["Operator_K", "Operator_L", "Operator_M"], "Area_F": ["Operator_E", "Operator_F", "Operator_G"]}
OPERATOR_BANDS = {"Operator_A": ["n28", "n78"], "Operator_B": ["n78"], "Operator_C": ["n78"], "Operator_D": ["n28"], "Operator_E": ["n77", "n260"], "Operator_F": ["n77", "n260"], "Operator_G": ["n41", "n258"], "Operator_H": ["n78", "n79", "n257"], "Operator_I": ["n78", "n258"], "Operator_J": ["n28", "n78"], "Operator_K": ["n28", "n78"], "Operator_L": ["n28", "n78"], "Operator_M": ["n78"]}
INFRASTRUCTURE_PROFILES = {"Nominal": 0.0, "Moderately_Degraded": 0.8, "Severely_Degraded": 1.5}

UE_PROFILES = {"UE-Advanced": {"tier": "Advanced", "throughput_efficiency": 1.10, "energy_efficiency": 0.95}, "UE-Standard": {"tier": "Standard", "throughput_efficiency": 1.00, "energy_efficiency": 1.00}, "UE-Basic": {"tier": "Basic", "throughput_efficiency": 0.85, "energy_efficiency": 1.10}}
UE_PROFILE_NAMES = list(UE_PROFILES.keys())
BAND_GAIN_DB = {"N/A": 0, "n28": 3, "n41": -2, "n77": 0, "n78": 0, "n79": -1, "n257": -13, "n258": -12, "n260": -14}

DOWNLINK_DEMAND_MEDIAN_MBPS = {"4G": {"Browse": 25, "Call": 4, "Gaming": 55, "Streaming": 120}, "5G NSA": {"Browse": 60, "Call": 8, "Gaming": 140, "Streaming": 350}, "5G SA": {"Browse": 70, "Call": 8, "Gaming": 170, "Streaming": 420}}
UPLINK_DEMAND_MEDIAN_MBPS = {"4G": {"Browse": 1.0, "Call": 0.5, "Gaming": 6, "Streaming": 20}, "5G NSA": {"Browse": 0.6, "Call": 0.3, "Gaming": 6, "Streaming": 35}, "5G SA": {"Browse": 1.0, "Call": 0.4, "Gaming": 8, "Streaming": 40}}
UPLINK_CAPACITY_RATIO = {"4G": 0.15, "5G NSA": 0.25, "5G SA": 0.35}

MOBILITY_TRANSITION_MATRIX = {"Static": {"states": ["Static", "Walking"], "weights": [0.5, 0.5]}, "Walking": {"states": ["Static", "Walking", "Driving"], "weights": [0.4, 0.4, 0.2]}, "Driving": {"states": ["Walking", "Driving"], "weights": [0.3, 0.7]}}
