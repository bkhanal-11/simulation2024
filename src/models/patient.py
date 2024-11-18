from dataclasses import dataclass
from enum import Enum


class PatientPriority(Enum):
    EMERGENCY = 1
    REGULAR = 2


@dataclass
class Patient:
    """Patient class with priority and blocking time tracking"""

    id: int
    arrival_time: float
    prep_time: float
    operation_time: float
    recovery_time: float
    priority: PatientPriority

    # for throughput
    prep_queue_entry: float = 0
    prep_start: float = 0
    prep_end: float = 0
    operation_start: float = 0
    operation_end: float = 0
    recovery_start: float = 0
    departure_time: float = 0

    def get_total_wait_time(self) -> float:
        """Calculate total waiting time in queues"""
        prep_wait = self.prep_start - self.prep_queue_entry
        operation_wait = self.operation_start - self.prep_end
        recovery_wait = self.recovery_start - self.operation_end
        return prep_wait + operation_wait + recovery_wait

    def get_blocking_time(self) -> float:
        """Calculate time OT was blocked waiting for recovery"""
        return self.recovery_start - self.operation_end

    def get_throughput_time(self) -> float:
        """Calculate total time in system"""
        return self.departure_time - self.arrival_time
