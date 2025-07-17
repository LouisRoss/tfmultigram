import sys

class Settings:
    history_length = 10
    max_token_strength = 20
    max_token_size = 20
    max_connected_tokens = 100
    decay_from_previous_activity = 2.0
    max_tokens = 2048
    null_distance = 0.5
    null_distance_dead = 0.2
    max_flux_per_connection = 0.5
    max_flux_total = 2.5
    distance_attenuation_function = [
        1.000, 0.900, 0.800, 0.700, 0.600,
        0.500, 0.400, 0.300, 0.025, 0.020,
        0.015, 0.010, 0.009, 0.008, 0.007,
        0.006, 0.005, 0.004, 0.004, 0.003,
        0.003, 0.002, 0.002, 0.001, 0.001
    ]

class MultigramState(sys.enum):
    """
    Enum for the state of the MultiGram.
    """
    IDLE = 0
    TRAINING = 1
    ANTICIPATING = 2
