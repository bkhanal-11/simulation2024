from dataclasses import dataclass
import simpy
import random
import statistics
from typing import List, Dict
from enum import Enum


class PatientPriority(Enum):
    EMERGENCY = 1
    REGULAR = 2


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


@dataclass
class Patient:
    """Patient class with priority and blocking time tracking"""

    id: int
    arrival_time: float
    prep_time: float
    operation_time: float
    recovery_time: float
    priority: PatientPriority

    # Timestamps for throughput calculation
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


class Hospital:
    """Main hospital simulation class with priority handling and blocking time"""

    def __init__(self, env: simpy.Environment, config: SimulationConfig):
        self.env = env
        self.config = config
        self.patient_completed_callback = None

        # Resource pools
        self.prep_rooms = simpy.PriorityResource(env, capacity=config.NUM_PREP_ROOMS)
        self.operating_theatre = simpy.PriorityResource(
            env, capacity=config.NUM_OPERATION_ROOMS
        )
        self.recovery_rooms = simpy.PriorityResource(
            env, capacity=config.NUM_RECOVERY_ROOMS
        )

        self.accepted_patient_count = 0
        self.current_blocking_start = None
        self.discarded_patients = 0

    def generate_patient(self) -> Patient:
        """Creates a new patient with priority level"""
        self.accepted_patient_count += 1

        is_emergency = random.random() < self.config.EMERGENCY_PROBABILITY
        priority = (
            PatientPriority.EMERGENCY if is_emergency else PatientPriority.REGULAR
        )

        prep_time_factor = (
            self.config.EMERGENCY_PREP_TIME_FACTOR if is_emergency else 1.0
        )
        operation_time_factor = (
            self.config.EMERGENCY_OPERATION_TIME_FACTOR if is_emergency else 1.0
        )

        return Patient(
            id=self.accepted_patient_count,
            arrival_time=self.env.now,
            prep_time=random.expovariate(
                1.0 / (self.config.MEAN_PREP_TIME * prep_time_factor)
            ),
            operation_time=random.expovariate(
                1.0 / (self.config.MEAN_OPERATION_TIME * operation_time_factor)
            ),
            recovery_time=random.expovariate(1.0 / self.config.MEAN_RECOVERY_TIME),
            priority=priority,
        )

    def patient_generator(self):
        """Generates new patients with priorities"""
        while True:
            yield self.env.timeout(
                random.expovariate(1.0 / self.config.MEAN_INTERARRIVAL_TIME)
            )

            if len(self.prep_rooms.queue) >= self.config.MAX_PREP_QUEUE_LENGTH:
                self.discarded_patients += 1
                continue

            patient = self.generate_patient()
            self.env.process(self.patient_process(patient))

    def patient_process(self, patient: Patient):
        """Handles individual patient's journey"""
        # preparation
        patient.prep_queue_entry = self.env.now
        prep_req = self.prep_rooms.request(priority=patient.priority.value)
        yield prep_req
        patient.prep_start = self.env.now
        yield self.env.timeout(patient.prep_time)
        patient.prep_end = self.env.now

        # operation
        operation_req = self.operating_theatre.request(priority=patient.priority.value)
        yield operation_req
        # since we got the operation room, we can release the prep room
        self.prep_rooms.release(prep_req)
        patient.operation_start = self.env.now
        yield self.env.timeout(patient.operation_time)
        patient.operation_end = self.env.now

        # recovery
        recovery_req = self.recovery_rooms.request(priority=patient.priority.value)
        yield recovery_req
        self.operating_theatre.release(operation_req)
        patient.recovery_start = self.env.now
        yield self.env.timeout(patient.recovery_time)
        self.recovery_rooms.release(recovery_req)
        patient.departure_time = self.env.now

        if self.patient_completed_callback:
            self.patient_completed_callback(patient)


class HospitalMonitor:
    """Enhanced monitor with priority statistics and blocking time"""

    def __init__(
        self, env: simpy.Environment, hospital: Hospital, interval: float = 1.0
    ):
        self.env = env
        self.hospital = hospital
        self.interval = interval

        self.hospital.patient_completed_callback = self.add_completed_patient

        self.emergency_patients = []
        self.regular_patients = []
        self.prep_utilization = []
        self.ot_utilization = []
        self.recovery_utilization = []

        self.emergency_patient_lengths = []
        self.regular_patient_lengths = []

        self.env.process(self.monitor())

    def monitor(self):
        """Monitor process with priority queue tracking"""
        while True:
            # Get queue lengths and split by priority
            prep_queue = len(self.hospital.prep_rooms.queue)
            emergency_prep = sum(
                1
                for req in self.hospital.prep_rooms.queue
                if req.priority == PatientPriority.EMERGENCY.value
            )
            regular_prep = prep_queue - emergency_prep

            self.emergency_patient_lengths.append(emergency_prep)
            self.regular_patient_lengths.append(regular_prep)

            # Track resource utilization
            self.prep_utilization.append(
                len(self.hospital.prep_rooms.users) / self.hospital.prep_rooms.capacity
            )
            self.ot_utilization.append(len(self.hospital.operating_theatre.users))
            self.recovery_utilization.append(
                len(self.hospital.recovery_rooms.users)
                / self.hospital.recovery_rooms.capacity
            )

            yield self.env.timeout(self.interval)

    def add_completed_patient(self, patient: Patient):
        """Record statistics by priority"""
        if patient.priority == PatientPriority.EMERGENCY:
            self.emergency_patients.append(patient)
        else:
            self.regular_patients.append(patient)

    def get_statistics(self) -> Dict:
        """Returns statistics separated by priority"""
        stats = {}

        total_blocking_time = 0
        all_patients = self.emergency_patients + self.regular_patients

        for priority, patients in [
            ("emergency", self.emergency_patients),
            ("regular", self.regular_patients),
        ]:
            if patients:
                waiting_times = [p.get_total_wait_time() for p in patients]
                throughput_times = [p.get_throughput_time() for p in patients]
                blocking_times = [p.get_blocking_time() for p in patients]

                stats[priority] = {
                    "count": len(patients),
                    "avg_waiting_time": statistics.mean(waiting_times),
                    "max_waiting_time": max(waiting_times),
                    "avg_throughput_time": statistics.mean(throughput_times),
                    "max_throughput_time": max(throughput_times),
                    "total_blocking_time": sum(blocking_times),
                    "avg_blocking_time": statistics.mean(blocking_times)
                    if blocking_times
                    else 0,
                }
                total_blocking_time += sum(blocking_times)

        # Overall system statistics
        stats["system"] = {
            "prep_utilization": statistics.mean(self.prep_utilization) * 100,
            "ot_utilization": statistics.mean(self.ot_utilization) * 100,
            "recovery_utilization": statistics.mean(self.recovery_utilization) * 100,
            "avg_emergency_queue": statistics.mean(self.emergency_patient_lengths),
            "avg_regular_queue": statistics.mean(self.regular_patient_lengths),
            "total_patients": len(all_patients),
            "total_blocking_time": total_blocking_time,
            "blocking_percentage": (total_blocking_time / self.env.now) * 100
            if self.env.now > 0
            else 0,
            "discarded_patients": self.hospital.discarded_patients,
        }

        return stats


def run_simulation(config: SimulationConfig = SimulationConfig()) -> Dict:
    """Runs the simulation with priority handling"""
    env = simpy.Environment()
    hospital = Hospital(env, config)
    monitor = HospitalMonitor(env, hospital)

    env.process(hospital.patient_generator())
    env.run(until=config.SIM_TIME)

    stats = monitor.get_statistics()

    print("\nSimulation Results with Priority Queue:")
    print("\nEmergency Patients:")
    if "emergency" in stats:
        print(f"Count: {stats['emergency']['count']}")
        print(f"Average Wait Time: {stats['emergency']['avg_waiting_time']:.2f}")
        print(
            f"Average Throughput Time: {stats['emergency']['avg_throughput_time']:.2f}"
        )
        print(f"Total Blocking Time: {stats['emergency']['total_blocking_time']:.2f}")

    print("\nRegular Patients:")
    if "regular" in stats:
        print(f"Count: {stats['regular']['count']}")
        print(f"Average Wait Time: {stats['regular']['avg_waiting_time']:.2f}")
        print(f"Average Throughput Time: {stats['regular']['avg_throughput_time']:.2f}")
        print(f"Total Blocking Time: {stats['regular']['total_blocking_time']:.2f}")

    print("\nSystem Statistics:")
    print(f"Operating Theatre Utilization: {stats['system']['ot_utilization']:.2f}%")
    print(
        f"Average Emergency Queue Length: {stats['system']['avg_emergency_queue']:.2f}"
    )
    print(f"Average Regular Queue Length: {stats['system']['avg_regular_queue']:.2f}")
    print(f"Total System Blocking Time: {stats['system']['total_blocking_time']:.2f}")
    print(f"Blocking Time Percentage: {stats['system']['blocking_percentage']:.2f}%")
    print(f"Discarded Patients: {stats['system']['discarded_patients']}")

    return stats


if __name__ == "__main__":
    results = run_simulation()
