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
        self.interarrival_distributions = [
            25.0,           # This is for exponential(25)
            22.5,           # This is for exponential(22.5)
            (20.0, 30.0),   # This for Uniform distribution(20,30)
            (20.0, 25.0)    # This for Uniform distribution(20,25)
        ]
        
        self.prep_time_distributions = [
            40.0,
            (30.0, 50.0)
        ]
        
        self.recovery_time_distributions = [
            40.0,
            (30.0, 50.0)
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
                NUM_OPERATION_ROOMS=1,  # Fixed at 1
                MEAN_OPERATION_TIME=20.0,  # Fixed with exp(20)
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
            
            # We want to collect samples at specified intervals for serial correlation
            samples = []
            def sample_collector():
                while True:
                    samples.append(len(hospital.prep_rooms.queue))
                    yield env.timeout(sample_interval)
            
            env.process(hospital.patient_generator())
            env.process(sample_collector())
            env.run(until=sample_interval * num_samples)
            
            all_series.append(samples)
        
        # For conveneince, we convert to DataFrame for correlation analysis
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
        
        # Creating dummy variables for categorical variables
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

def prepare_prediction_data(
    interarrival_type: str,
    interarrival_mean: float,
    prep_type: str,
    prep_mean: float,
    num_prep_rooms: int,
    num_recovery_rooms: int
) -> pd.DataFrame:
    """
    Prepare data for prediction in the format expected by the model
    """
    data = pd.DataFrame({
        'interarrival_type_uniform': [1 if interarrival_type == 'uniform' else 0],
        'prep_type_uniform': [1 if prep_type == 'uniform' else 0],
        'interarrival_mean': [float(interarrival_mean)],
        'prep_mean': [float(prep_mean)],
        'num_prep_rooms': [float(num_prep_rooms)],
        'num_recovery_rooms': [float(num_recovery_rooms)]
    })
    
    return sm.add_constant(data)

def predict_with_intervals(model, X_new):
    """
    Make prediction with confidence intervals
    """
    prediction = model.get_prediction(X_new)
    intervals = prediction.conf_int(alpha=0.05)  # 95% confidence interval
    mean_prediction = prediction.predicted_mean
    
    return {
        'prediction': mean_prediction[0],
        'lower_bound': intervals[0][0],
        'upper_bound': intervals[0][1]
    }


if __name__ == "__main__":
    exp_design = ExperimentDesign()

    configs = exp_design.generate_fractional_factorial()

    correlation_df = exp_design.analyze_serial_correlation(configs[0])

    results_df = exp_design.run_experiments(configs)

    model = exp_design.build_regression_model(results_df)
    print(model.summary())
    
    print("\n================================================================")
    print("|                    Prediction Examples                       |")
    print("================================================================")

    # Prediction examples
    scenarios = [
        {
            'interarrival_type': 'exponential',
            'interarrival_mean': 25.0,
            'prep_type': 'exponential',
            'prep_mean': 40.0,
            'num_prep_rooms': 4,
            'num_recovery_rooms': 4
        },
        {
            'interarrival_type': 'uniform',
            'interarrival_mean': 25.0,
            'prep_type': 'uniform',
            'prep_mean': 40.0,
            'num_prep_rooms': 5,
            'num_recovery_rooms': 5
        }
    ]

    # Make predictions
    for scenario in scenarios:
        X_new = prepare_prediction_data(**scenario)
        predicted_queue_length = model.predict(X_new)
        
        print(f"\nScenario:")
        for key, value in scenario.items():
            print(f"{key}: {value}")
        print(f"\nPredicted queue length: {predicted_queue_length[0]:.2f}")
    
    # Example with confidence intervals
    X_new = prepare_prediction_data(
        interarrival_type='exponential',
        interarrival_mean=25.0,
        prep_type='exponential',
        prep_mean=40.0,
        num_prep_rooms=4,
        num_recovery_rooms=4
    )

    result = predict_with_intervals(model, X_new)
    print("\nPrediction with confidence intervals:")
    print(f"\nPredicted queue length: {result['prediction']:.2f}")
    print(f"95% Confidence Interval: ({result['lower_bound']:.2f}, {result['upper_bound']:.2f})")
