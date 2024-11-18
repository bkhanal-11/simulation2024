import simpy
import statistics
from typing import Dict
from models.patient import Patient, PatientPriority
from simulation.hospital import Hospital


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
