# TIES481-24 Hospital Simulation 🏥

A simple hospital simulation system for operating theaters and patient queues 👨‍⚕️.

## Requirements ⚡

Install Python packages with this command.

```bash
pip install -r requirements.txt
```

## How to Run 🚀

1. Enter the src folder.

```bash
cd src
```

2. Run this command to start the simulation.

```bash
python main.py
```

## Project Structure 📂

```bash
.
├── docs
│   ├── main.tex
│   └── static
├── README.md
├── requirements.txt
└── src
    ├── config.json
    ├── main.py
    ├── models
    └── simulation
```

### Folder Descriptions

- **docs/**: Contains documentation files, including LaTeX source files related to Assignment 1.
- **README.md**: The main readme file providing an overview and instructions for the project.
- **requirements.txt**: Lists the Python packages required to run the simulation.
- **src/**: The source code directory containing the main application logic.
  - **config.json**: Configuration file for setting up simulation parameters.
  - **main.py**: The entry point for running the simulation.
  - **models/**: Contains the data models used in the simulation.
  - **simulation/**: Includes the core simulation logic and algorithms.

## Implementation Model Details 🛠️

The hospital simulation is designed to model the flow of patients through a hospital system, focusing on the use of operating theaters and patient queues. The simulation is built using the `simpy` library, which provides a framework for discrete-event simulation.

### Core Components

- **SimulationConfig (src/models/config.py)**: Defines the configuration parameters for the simulation, such as mean times for interarrival, preparation, operation, and recovery, as well as the number of rooms and emergency handling factors.

- **Patient (src/models/patient.py)**: Represents a patient in the simulation, including attributes for arrival, preparation, operation, and recovery times, as well as priority (emergency or regular). Methods are provided to calculate total wait time, blocking time, and throughput time.

- **Hospital (src/simulation/hospital.py)**: The main class that simulates the hospital environment. It manages resources like preparation rooms, operating theaters, and recovery rooms using priority queues. It generates patients and processes them through the hospital system.

- **HospitalMonitor (src/simulation/monitor.py)**: Monitors the simulation, collecting statistics on patient flow and resource utilization. It tracks queue lengths, resource usage, and patient statistics by priority.

### Simulation Flow

1. **Patient Generation**: Patients are generated at random intervals based on the mean interarrival time. Each patient is assigned a priority (emergency or regular) and corresponding preparation and operation times.

2. **Patient Processing**: Patients go through preparation, operation, and recovery stages. Each stage uses a priority resource to manage access, ensuring that emergency patients are prioritized.

3. **Monitoring and Statistics**: The monitor collects data on queue lengths, resource utilization, and patient statistics. It provides detailed output on the performance of the hospital system, including average wait times, throughput times, and blocking times.
