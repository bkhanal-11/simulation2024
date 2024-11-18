import json
import simpy
from typing import Dict
from models.config import SimulationConfig
from simulation.hospital import Hospital
from simulation.monitor import HospitalMonitor


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
    with open("config.json", "r") as f:
        config_dict = json.load(f)

    config = SimulationConfig(**config_dict)

    _ = run_simulation(config)
