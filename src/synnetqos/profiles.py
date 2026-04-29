"""Reference synthetic scenario profiles.

Use Deployment_Area labels rather than real city names in the public dataset.
Coordinates are synthetic centers used to create spatially coherent session logs.
They are not empirical measurements of real cities.
"""

DEPLOYMENT_AREAS = {
    "Area_A": (19.0760, 72.8777, "Urban"),
    "Area_B": (28.6139, 77.2090, "Urban"),
    "Area_C": (40.7128, -74.0060, "Urban"),
    "Area_D": (35.6895, 139.6917, "Urban"),
    "Area_E": (52.5200, 13.4050, "Urban"),
    "Area_F": (37.7749, -122.4194, "Urban"),
}

OPERATOR_AVAILABILITY = {
    "Area_A": ["Operator_A", "Operator_B", "Operator_C"],
    "Area_B": ["Operator_A", "Operator_B", "Operator_D"],
    "Area_C": ["Operator_E", "Operator_F", "Operator_G"],
    "Area_D": ["Operator_H", "Operator_I", "Operator_J"],
    "Area_E": ["Operator_K", "Operator_L", "Operator_M"],
    "Area_F": ["Operator_E", "Operator_F", "Operator_G"],
}

OPERATOR_BANDS = {
    "Operator_A": ["n28", "n78"],
    "Operator_B": ["n78"],
    "Operator_C": ["n78"],
    "Operator_D": ["n28"],
    "Operator_E": ["n77", "n260"],
    "Operator_F": ["n77", "n260"],
    "Operator_G": ["n41", "n258"],
    "Operator_H": ["n78", "n79", "n257"],
    "Operator_I": ["n78", "n258"],
    "Operator_J": ["n28", "n78"],
    "Operator_K": ["n28", "n78"],
    "Operator_L": ["n28", "n78"],
    "Operator_M": ["n78"],
}

INFRASTRUCTURE_PROFILES = {
    "Nominal": 0.0,
    "Moderately_Degraded": 0.8,
    "Severely_Degraded": 1.5,
}

UE_PROFILES = {
    "UE-Advanced": {"class": "Advanced", "throughput_efficiency": 1.10, "energy_efficiency": 0.95},
    "UE-Standard": {"class": "Standard", "throughput_efficiency": 1.00, "energy_efficiency": 1.00},
    "UE-Basic": {"class": "Basic", "throughput_efficiency": 0.85, "energy_efficiency": 1.10},
}
