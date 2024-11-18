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
    NUM_RECOVERY_ROOMS: int = 3
    SIM_TIME: float = 10000
    
    # Parameters for emergency cases
    EMERGENCY_PROBABILITY: float = 0.2  # 20% chance of emergency
    EMERGENCY_PREP_TIME_FACTOR: float = 0.5  # Emergency prep takes half the time
    EMERGENCY_OPERATION_TIME_FACTOR: float = 0.8  # Emergency operation takes 80% of regular time

@dataclass
class Patient:
    """Patient class with priority"""
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
    operation_queue_entry: float = 0
    operation_start: float = 0
    operation_end: float = 0
    recovery_queue_entry: float = 0
    recovery_start: float = 0
    departure_time: float = 0

    def get_total_wait_time(self) -> float:
        """Calculate total waiting time in queues"""
        prep_wait = self.prep_start - self.prep_queue_entry
        operation_wait = self.operation_start - self.prep_end
        recovery_wait = self.recovery_start - self.operation_end
        return prep_wait + operation_wait + recovery_wait

    def get_throughput_time(self) -> float:
        """Calculate total time in system"""
        return self.departure_time - self.arrival_time

class Hospital:
    """Main hospital simulation class with priority handling"""
    def __init__(self, env: simpy.Environment, config: SimulationConfig):
        self.env = env
        self.config = config
        self.monitor = None
        
        # Resource pools
        self.prep_rooms = simpy.PriorityResource(env, capacity=config.NUM_PREP_ROOMS)
        self.operating_theatre = simpy.PriorityResource(env, capacity=1)
        self.recovery_rooms = simpy.PriorityResource(env, capacity=config.NUM_RECOVERY_ROOMS)
        
        self.patient_count = 0
        
    def generate_patient(self) -> Patient:
        """Creates a new patient with priority level"""
        self.patient_count += 1
        
        # Determine if this is an emergency patient
        is_emergency = random.random() < self.config.EMERGENCY_PROBABILITY
        priority = PatientPriority.EMERGENCY if is_emergency else PatientPriority.REGULAR
        
        # Adjust service times based on priority
        prep_time_factor = self.config.EMERGENCY_PREP_TIME_FACTOR if is_emergency else 1.0
        operation_time_factor = self.config.EMERGENCY_OPERATION_TIME_FACTOR if is_emergency else 1.0
        
        return Patient(
            id=self.patient_count,
            arrival_time=self.env.now,
            prep_time=random.expovariate(1.0/(self.config.MEAN_PREP_TIME * prep_time_factor)),
            operation_time=random.expovariate(1.0/(self.config.MEAN_OPERATION_TIME * operation_time_factor)),
            recovery_time=random.expovariate(1.0/self.config.MEAN_RECOVERY_TIME),
            priority=priority
        )
    
    def patient_generator(self):
        """Generates new patients with priorities"""
        while True:
            yield self.env.timeout(random.expovariate(1.0/self.config.MEAN_INTERARRIVAL_TIME))
            patient = self.generate_patient()
            patient.prep_queue_entry = self.env.now
            self.env.process(self.patient_process(patient))
    
    def patient_process(self, patient: Patient):
        """Handles individual patient's journey with priority"""
        # Preparation phase
        with self.prep_rooms.request(priority=patient.priority.value) as req:
            yield req
            patient.prep_start = self.env.now
            yield self.env.timeout(patient.prep_time)
            patient.prep_end = self.env.now
        
        # Operation phase
        patient.operation_queue_entry = self.env.now
        with self.operating_theatre.request(priority=patient.priority.value) as req:
            yield req
            patient.operation_start = self.env.now
            yield self.env.timeout(patient.operation_time)
            patient.operation_end = self.env.now
        
        # Recovery phase
        patient.recovery_queue_entry = self.env.now
        with self.recovery_rooms.request(priority=patient.priority.value) as req:
            yield req
            patient.recovery_start = self.env.now
            yield self.env.timeout(patient.recovery_time)
            patient.departure_time = self.env.now
            
            if self.monitor:
                self.monitor.add_completed_patient(patient)

class HospitalMonitor:
    """Enhanced monitor with priority statistics"""
    def __init__(self, env: simpy.Environment, hospital: 'Hospital', interval: float = 1.0):
        self.env = env
        self.hospital = hospital
        self.interval = interval
        
        self.emergency_patients = []
        self.regular_patients = []
        self.prep_utilization = []
        self.ot_utilization = []
        self.recovery_utilization = []
        
        self.emergency_queue_lengths = []
        self.regular_queue_lengths = []
        
        self.env.process(self.monitor())
    
    def monitor(self):
        """Monitor process with priority queue tracking"""
        while True:
            # Get queue lengths and split by priority
            prep_queue = len(self.hospital.prep_rooms.queue)
            emergency_prep = sum(1 for req in self.hospital.prep_rooms.queue 
                               if req.priority == PatientPriority.EMERGENCY.value)
            regular_prep = prep_queue - emergency_prep
            
            self.emergency_queue_lengths.append(emergency_prep)
            self.regular_queue_lengths.append(regular_prep)
            
            # Track resource utilization
            self.prep_utilization.append(len(self.hospital.prep_rooms.users) / self.hospital.prep_rooms.capacity)
            self.ot_utilization.append(len(self.hospital.operating_theatre.users))
            self.recovery_utilization.append(len(self.hospital.recovery_rooms.users) / self.hospital.recovery_rooms.capacity)
            
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
        
        for priority, patients in [
            ("emergency", self.emergency_patients),
            ("regular", self.regular_patients)
        ]:
            if patients:
                waiting_times = [p.get_total_wait_time() for p in patients]
                throughput_times = [p.get_throughput_time() for p in patients]
                
                stats[priority] = {
                    'count': len(patients),
                    'avg_waiting_time': statistics.mean(waiting_times),
                    'max_waiting_time': max(waiting_times),
                    'avg_throughput_time': statistics.mean(throughput_times),
                    'max_throughput_time': max(throughput_times)
                }
        
        # Overall system statistics
        stats['system'] = {
            'prep_utilization': statistics.mean(self.prep_utilization) * 100,
            'ot_utilization': statistics.mean(self.ot_utilization) * 100,
            'recovery_utilization': statistics.mean(self.recovery_utilization) * 100,
            'avg_emergency_queue': statistics.mean(self.emergency_queue_lengths),
            'avg_regular_queue': statistics.mean(self.regular_queue_lengths),
            'total_patients': len(self.emergency_patients) + len(self.regular_patients)
        }
        
        return stats

def run_simulation(config: SimulationConfig = SimulationConfig()) -> Dict:
    """Runs the simulation with priority handling"""
    env = simpy.Environment()
    hospital = Hospital(env, config)
    monitor = HospitalMonitor(env, hospital)
    hospital.monitor = monitor
    
    env.process(hospital.patient_generator())
    env.run(until=config.SIM_TIME)
    
    stats = monitor.get_statistics()
    
    print("\nSimulation Results with Priority Queue:")
    print(f"\nEmergency Patients:")
    if 'emergency' in stats:
        print(f"Count: {stats['emergency']['count']}")
        print(f"Average Wait Time: {stats['emergency']['avg_waiting_time']:.2f}")
        print(f"Average Throughput Time: {stats['emergency']['avg_throughput_time']:.2f}")
    
    print(f"\nRegular Patients:")
    if 'regular' in stats:
        print(f"Count: {stats['regular']['count']}")
        print(f"Average Wait Time: {stats['regular']['avg_waiting_time']:.2f}")
        print(f"Average Throughput Time: {stats['regular']['avg_throughput_time']:.2f}")
    
    print(f"\nSystem Statistics:")
    print(f"Operating Theatre Utilization: {stats['system']['ot_utilization']:.2f}%")
    print(f"Average Emergency Queue Length: {stats['system']['avg_emergency_queue']:.2f}")
    print(f"Average Regular Queue Length: {stats['system']['avg_regular_queue']:.2f}")
    
    return stats

if __name__ == "__main__":
    results = run_simulation()