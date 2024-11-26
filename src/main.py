import simpy
import random
import numpy as np
from typing import Dict, List, Tuple
from models.config import SimulationConfig
from simulation.hospital import Hospital
from simulation.monitor import HospitalMonitor
from scipy import stats
import json

def run_simulation(config: SimulationConfig, seed: int, warmup_time: float = 1000.0) -> Dict:
    """Run a single simulation with warmup period"""
    np.random.seed(seed)
    random.seed(seed)
    env = simpy.Environment()
    hospital = Hospital(env, config)
    monitor = HospitalMonitor(env, hospital)

    # run the simulation until warmup
    env.process(hospital.patient_generator())
    env.run(until=warmup_time)

    monitor.emergency_patient_lengths = []
    monitor.regular_patient_lengths = []
    monitor.prep_utilization = []
    monitor.ot_utilization = []
    monitor.recovery_utilization = []
    
    # continue simulation for simualtion time
    env.run(until=warmup_time + config.SIM_TIME)
    
    return monitor.get_statistics()

def compute_confidence_intervals(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and confidence interval with bounds clamping for percentages"""
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    
    # to calculate percentage metrics like blocking probability
    if all(0 <= x <= 100 for x in data):  # checks if the data represents percentages
        return (
            mean,
            max(0, mean - interval),  # lower bound
            min(100, mean + interval)  # upper bound
        )
    
    return mean, mean - interval, mean + interval

def print_metric(name: str, stats: Tuple[float, float, float], is_percentage: bool = False):
    """Format and print a metric with appropriate formatting"""
    mean, ci_low, ci_high = stats
    format_str = ".2f" if is_percentage else ".3f"
    suffix = "%" if is_percentage else ""
    print(f"{name}: {mean:{format_str}}{suffix} ({ci_low:{format_str}}{suffix}, {ci_high:{format_str}}{suffix})")

def analyze_differences(results: Dict) -> Dict:
    """Analyze differences between configurations"""
    differences = {}
    configs = list(results.keys())
    
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            config1, config2 = configs[i], configs[j]
            key = f"{config1}_vs_{config2}"
            
            # Calculate paired differences for each metric
            queue_diffs = []
            blocking_diffs = []
            for k in range(len(results[config1])):
                stats1 = results[config1][k]
                stats2 = results[config2][k]
                
                queue_diffs.append(
                    stats1['system']['avg_regular_queue'] - 
                    stats2['system']['avg_regular_queue']
                )
                blocking_diffs.append(
                    stats1['system']['blocking_percentage'] - 
                    stats2['system']['blocking_percentage']
                )
            
            differences[key] = {
                'queue_length': compute_confidence_intervals(queue_diffs),
                'blocking_probability': compute_confidence_intervals(blocking_diffs)
            }
    
    return differences

def main():    
    # Define configurations
    configs = [
        SimulationConfig(NUM_PREP_ROOMS=3, NUM_RECOVERY_ROOMS=4, SIM_TIME=1000),
        SimulationConfig(NUM_PREP_ROOMS=3, NUM_RECOVERY_ROOMS=5, SIM_TIME=1000),
        SimulationConfig(NUM_PREP_ROOMS=4, NUM_RECOVERY_ROOMS=5, SIM_TIME=1000)
    ]
    
    num_samples = 20
    results = {str(config): [] for config in configs}
    
    # Run simulations
    for config in configs:
        for seed in range(num_samples):
            stats = run_simulation(config, seed)
            results[str(config)].append(stats)
    
    # Compute statistics for each configuration
    for config, stats_list in results.items():
        queue_lengths = [stats['system']['avg_regular_queue'] for stats in stats_list]
        blocking_probs = [stats['system']['blocking_percentage'] for stats in stats_list]
        recovery_utils = [stats['system']['recovery_utilization'] for stats in stats_list]

        print(f"\nConfiguration: {config}")
        print_metric("Queue Length", compute_confidence_intervals(queue_lengths))
        print_metric("Blocking Probability", compute_confidence_intervals(blocking_probs), is_percentage=True)
        print_metric("Recovery Facilities Utilization", compute_confidence_intervals(recovery_utils), is_percentage=True)
    
    # Analyze paired differences
    differences = analyze_differences(results)
    print("\nPaired Differences:")
    for comparison, metrics in differences.items():
        print(f"\n{comparison}:")
        for metric, (mean, ci_low, ci_high) in metrics.items():
            is_percentage = 'probability' in metric.lower()
            significant = ci_low * ci_high > 0  # Both CI bounds have same sign
            print_metric(
                f"{metric}", 
                (mean, ci_low, ci_high), 
                is_percentage=is_percentage
            )
            if significant:
                print("*Significant*")

if __name__ == "__main__":
    main()