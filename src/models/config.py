from dataclasses import dataclass
from typing import Tuple, Union
import random

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation"""

    MEAN_INTERARRIVAL_TIME: Union[float, Tuple[float, float]] = 25.0
    MEAN_PREP_TIME: Union[float, Tuple[float, float]] = 40.0
    MEAN_OPERATION_TIME: float = 20.0
    MEAN_RECOVERY_TIME: Union[float, Tuple[float, float]] = 40.0
    NUM_PREP_ROOMS: int = 3
    NUM_OPERATION_ROOMS: int = 1
    NUM_RECOVERY_ROOMS: int = 3
    SIM_TIME: float = 10000

    EMERGENCY_PROBABILITY: float = 0.2
    EMERGENCY_PREP_TIME_FACTOR: float = 0.5
    EMERGENCY_OPERATION_TIME_FACTOR: float = 0.8
    MAX_PREP_QUEUE_LENGTH: int = 4

    def get_interarrival_time(self) -> float:
        if isinstance(self.MEAN_INTERARRIVAL_TIME, tuple):
            return random.uniform(*self.MEAN_INTERARRIVAL_TIME)
        return random.expovariate(1.0 / self.MEAN_INTERARRIVAL_TIME)

    def get_prep_time(self, is_emergency: bool) -> float:
        factor = self.EMERGENCY_PREP_TIME_FACTOR if is_emergency else 1.0
        if isinstance(self.MEAN_PREP_TIME, tuple):
            return random.uniform(*self.MEAN_PREP_TIME) * factor
        return random.expovariate(1.0 / (self.MEAN_PREP_TIME * factor))

    def get_operation_time(self, is_emergency: bool) -> float:
        factor = self.EMERGENCY_OPERATION_TIME_FACTOR if is_emergency else 1.0
        return random.expovariate(1.0 / (self.MEAN_OPERATION_TIME * factor))

    def get_recovery_time(self) -> float:
        if isinstance(self.MEAN_RECOVERY_TIME, tuple):
            return random.uniform(*self.MEAN_RECOVERY_TIME)
        return random.expovariate(1.0 / self.MEAN_RECOVERY_TIME)