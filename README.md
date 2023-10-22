# Data Center Environment and Reinforcement Learning Control

This repository offers a comprehensive simulation of a data center (DC) environment using EnergyPlus, accompanied by a state-of-the-art Reinforcement Learning agent trained for dynamic cooling and humidity control within the data center.

### Key Features:

- **Data Center Model**: A detailed model of a data center that includes hot and cold aisles, raised floor, airflow management, cooling systems, and temperature-sensitive zones. It is built using EnergyPlus, a robust building energy simulation engine.

- **Reinforcement Learning Agent (PPO)**: An RL agent based on Proximal Policy Optimization (PPO) is trained to autonomously manage and optimize cooling and humidity control within the data center. It learns how to respond to changing conditions and maintain a stable, energy-efficient, and environment-friendly operation.

- **HTTP API Interface**: Provides a convenient HTTP-based API for interacting with the trained PPO agent. You can connect external applications or scripts to request actions and receive observations from the agent, making it easy to integrate with various control systems.

- **Comparison with Constant Control Agents**: This repository also includes comparative results between the PPO agent and constant control strategies. This allows you to assess the superiority of RL-based control in managing data center conditions efficiently.

- **Database Integration (ClickHouse)**: Code is provided to create a database and table in ClickHouse.

- **Superset Dashboard**: An archive with a Superset dashboard is available, providing intuitive visualizations and analytics for simulations or inference. Archive must be exported to superset

### Requirements:

- Python libraries mentioned in `requirements.txt`.

- EnergyPlus version 23-1-0 is required for the data center simulation.

To perform inference or analyze your simulation results, you'll need to set up [Superset](https://superset.apache.org/) and [ClickHouse](https://clickhouse.tech/). Please ensure that both are installed and running. If you haven't already set up these tools, you can find detailed instructions and example configurations in our [repository](https://github.com/YARIK-AI/ML).

### Usage

`make help` for view existing commands.

### Compatibility:

This repository is designed to work on various platforms, but extensive testing has been done primarily on MacOS. Windows compatibility may require additional adjustments.
