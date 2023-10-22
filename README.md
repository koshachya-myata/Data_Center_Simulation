# Data Center Environment and Reinforcement Learning (RL) Control

This repository offers a comprehensive simulation of a data center environment using EnergyPlus, accompanied by a state-of-the-art Reinforcement Learning agent trained for dynamic cooling and humidity control within the data center.

### Key Features:

- **Data Center Model**: A detailed simulation of a data center that includes hot and cold aisles, raised floor, airflow management, cooling systems, and temperature-sensitive zones. It is built using EnergyPlus, a robust building energy simulation engine.

- **Reinforcement Learning Agent (PPO)**: An RL agent based on Proximal Policy Optimization (PPO) is trained to autonomously manage and optimize cooling and humidity control within the data center. It learns how to respond to changing conditions and maintain a stable, energy-efficient, and environment-friendly operation.

- **HTTP API Interface**: Provides a convenient HTTP-based API for interacting with the trained PPO agent. You can connect external applications or scripts to request actions and receive observations from the agent, making it easy to integrate with various control systems.

- **Comparison with Constant Control Agents**: This repository also includes comparative results between the PPO agent and constant control strategies. This allows you to assess the superiority of RL-based control in managing data center conditions efficiently.

- **Database Integration (ClickHouse)**: Code is provided to create a database in ClickHouse. You can store and analyze data from your data center simulations over time, facilitating long-term performance analysis and decision-making.

- **Superset Dashboard**: An archive with a Superset dashboard is available, providing intuitive visualizations and analytics for your simulations. It's a valuable tool for monitoring and analyzing data center performance.

### Requirements:

- Python libraries mentioned in `requirements.txt`.

- EnergyPlus version 23-1-0 is required for the data center simulation.

### Compatibility:

This repository is designed to work on various platforms, but extensive testing has been done primarily on MacOS. Windows compatibility may require additional adjustments.

### Usage

`make help` for view existing commands.
