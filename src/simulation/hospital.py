import simpy
import random
from models.config import SimulationConfig
from models.patient import Patient, PatientPriority


class Hospital:
    """Main hospital simulation class"""

    def __init__(self, env: simpy.Environment, config: SimulationConfig):
        self.env = env
        self.config = config
        self.patient_completed_callback = None

        # resources
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
