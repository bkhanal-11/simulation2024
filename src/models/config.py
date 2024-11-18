from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation"""

    MEAN_INTERARRIVAL_TIME: float = 25.0
    MEAN_PREP_TIME: float = 40.0
    MEAN_OPERATION_TIME: float = 20.0
    MEAN_RECOVERY_TIME: float = 40.0
    NUM_PREP_ROOMS: int = 3
    NUM_OPERATION_ROOMS: int = 1
    NUM_RECOVERY_ROOMS: int = 3
    SIM_TIME: float = 10000

    EMERGENCY_PROBABILITY: float = 0.2
    EMERGENCY_PREP_TIME_FACTOR: float = 0.5
    EMERGENCY_OPERATION_TIME_FACTOR: float = 0.8
    MAX_PREP_QUEUE_LENGTH: int = 4
