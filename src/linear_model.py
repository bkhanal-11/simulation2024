import simpy
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import product
import random
from typing import List, Tuple, Union
from dataclasses import dataclass
from models.config import SimulationConfig
from simulation.hospital import Hospital
from simulation.monitor import HospitalMonitor

class ExperimentDesign:
    def __init__(self):
        # Define factor levels using Union[float, Tuple[float, float]] type
        self.interarrival_distributions = [
            25.0,           # exp(25)
            22.5,           # exp(22.5)
            (20.0, 30.0),   # Unif(20,30)
            (20.0, 25.0)    # Unif(20,25)
        ]
        
        self.prep_time_distributions = [
            40.0,           # exp(40)
            (30.0, 50.0)    # Unif(30,50)
        ]
        
        self.recovery_time_distributions = [
            40.0,           # exp(40)
            (30.0, 50.0)    # Unif(30,50)
        ]
        
        self.prep_units = [4, 5]
        self.recovery_units = [4, 5]

    def generate_fractional_factorial(self) -> List[SimulationConfig]:
        """Generate 2^(6-3) fractional factorial design"""
        # Basic factors (first 3)
        base_design = list(product([0, 1], repeat=3))
        
        configs = []
        for b1, b2, b3 in base_design:
            # Generate aliased factors (resolution III design)
            b4 = (b1 + b2) % 2
            b5 = (b2 + b3) % 2
            b6 = (b1 + b3) % 2
            
            config = SimulationConfig(
                MEAN_INTERARRIVAL_TIME=self.interarrival_distributions[b1 * 2],
                MEAN_PREP_TIME=self.prep_time_distributions[b2],
                MEAN_RECOVERY_TIME=self.recovery_time_distributions[b3],
                NUM_PREP_ROOMS=self.prep_units[b4],
                NUM_RECOVERY_ROOMS=self.recovery_units[b5],
                NUM_OPERATION_ROOMS=1,  # Fixed
                MEAN_OPERATION_TIME=20.0,  # Fixed exp(20)
                SIM_TIME=10000
            )
            configs.append(config)
        
        return configs

    def run_experiments(self, configs: List[SimulationConfig], 
                       num_replications: int = 10) -> pd.DataFrame:
        """Run experiments with multiple replications"""
        results = []
        
        for config in configs:
            for _ in range(num_replications):
                stats = self._run_single_simulation(config)
                
                # Record results
                result = {
                    'interarrival_type': 'uniform' if isinstance(config.MEAN_INTERARRIVAL_TIME, tuple) else 'exponential',
                    'interarrival_mean': (sum(config.MEAN_INTERARRIVAL_TIME)/2 
                                        if isinstance(config.MEAN_INTERARRIVAL_TIME, tuple)
                                        else config.MEAN_INTERARRIVAL_TIME),
                    'prep_type': 'uniform' if isinstance(config.MEAN_PREP_TIME, tuple) else 'exponential',
                    'prep_mean': (sum(config.MEAN_PREP_TIME)/2 
                                if isinstance(config.MEAN_PREP_TIME, tuple)
                                else config.MEAN_PREP_TIME),
                    'num_prep_rooms': config.NUM_PREP_ROOMS,
                    'num_recovery_rooms': config.NUM_RECOVERY_ROOMS,
                    'avg_queue_length': stats['system']['avg_regular_queue'],
                    'ot_utilization': stats['system']['ot_utilization']
                }
                results.append(result)
                
        return pd.DataFrame(results)

    def analyze_serial_correlation(self, config: SimulationConfig, 
                                 num_runs: int = 10, 
                                 num_samples: int = 10,
                                 sample_interval: float = 100) -> pd.DataFrame:
        """Analyze serial correlation in queue lengths"""
        all_series = []
        
        for run in range(num_runs):
            env = simpy.Environment()
            hospital = Hospital(env, config)
            monitor = HospitalMonitor(env, hospital)
            
            # Collect samples at specified intervals
            samples = []
            def sample_collector():
                while True:
                    samples.append(len(hospital.prep_rooms.queue))
                    yield env.timeout(sample_interval)
            
            env.process(hospital.patient_generator())
            env.process(sample_collector())
            env.run(until=sample_interval * num_samples)
            
            all_series.append(samples)
        
        # Convert to DataFrame for correlation analysis
        df = pd.DataFrame(all_series)
        return df

    def _run_single_simulation(self, config: SimulationConfig) -> dict:
        """Run a single simulation replication"""
        env = simpy.Environment()
        hospital = Hospital(env, config)
        monitor = HospitalMonitor(env, hospital)
        
        env.process(hospital.patient_generator())
        env.run(until=config.SIM_TIME)
        
        return monitor.get_statistics()

    def build_regression_model(self, results_df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
        """Build and analyze regression model with proper data type handling"""
        
        # Create dummy variables for categorical variables
        categorical_features = ['interarrival_type', 'prep_type']
        dummies = pd.get_dummies(results_df[categorical_features], drop_first=True)
        
        # Create feature matrix X with proper numeric types
        X = pd.DataFrame()
        
        # Add dummy variables
        for col in dummies.columns:
            X[col] = dummies[col].astype(float)
        
        # Add numeric variables
        X['interarrival_mean'] = results_df['interarrival_mean'].astype(float)
        X['prep_mean'] = results_df['prep_mean'].astype(float)
        X['num_prep_rooms'] = results_df['num_prep_rooms'].astype(float)
        X['num_recovery_rooms'] = results_df['num_recovery_rooms'].astype(float)
        
        # Add constant
        X = sm.add_constant(X)
        
        # Convert target variable to float
        y = results_df['avg_queue_length'].astype(float)
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        # Print feature names for verification
        print("\nFeatures included in regression:")
        for i, feature in enumerate(X.columns):
            print(f"{i+1}. {feature}")
        
        return model

if __name__ == "__main__":
    # Create experiment design instance
    exp_design = ExperimentDesign()

    # Generate experimental configurations
    configs = exp_design.generate_fractional_factorial()

    # Analyze serial correlation for a specific configuration
    correlation_df = exp_design.analyze_serial_correlation(configs[0])

    # Run the full experiment
    results_df = exp_design.run_experiments(configs)

    # Build and analyze regression model
    model = exp_design.build_regression_model(results_df)
    print(model.summary())