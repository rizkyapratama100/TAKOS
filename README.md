# TAKOS: Tentacle Actuation and Kinematic Operational Simulation

This project simulates a tendon-driven tentacle robot using MuJoCo and integrates it with ROS 2 for control and visualization. It includes Reinforcement Learning (RL) training scripts using Stable Baselines3.

## Onboarding Guide: WSL (Ubuntu 22.04) Setup

Follow these steps to set up the project on Windows Subsystem for Linux (WSL).

### 1. Prerequisites
*   **Windows 10/11** with WSL 2 installed.
*   **Ubuntu 22.04 (Jammy Jellyfish)** installed in WSL.
*   **X-Server for GUI (Optional but Recommended):** Install [GWSL](https://opticos.github.io/gwsl/) or use the built-in WSLg (Windows 11) to view MuJoCo and RViz windows.

### 2. Install ROS 2 Humble
Since this project uses `rclpy 3.3.x`, it is built for **ROS 2 Humble Hawksbill**.

```bash
# Set locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/index.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Base and Development Tools
sudo apt update
sudo apt install ros-humble-ros-base python3-colcon-common-extensions -y

# Source the ROS 2 environment (add to ~/.bashrc for convenience)
source /opt/ros/humble/setup.bash
```

### 3. Setup Python Environment
It is recommended to use a virtual environment.

```bash
# Install Python pip and venv
sudo apt install python3-pip python3-venv -y

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install project dependencies
pip install -r requirements.txt
```

### 4. Running the Simulation
To run the MuJoCo simulation with the ROS 2 interface:

```bash
# In one terminal (ensure ROS is sourced and venv is active)
python3 run_sim.py
```

### 5. Interacting with the Simulation
You can control the tentacle via ROS 2 topics:

```bash
# Control tendons (2D array)
ros2 topic pub --once /tentacle/tendon_cmds std_msgs/msg/Float64MultiArray "{data: [0.5, 0.5]}"

# Control joints (23D array)
ros2 topic pub --once /tentacle/tendon_cmds std_msgs/msg/Float64MultiArray "{data: [0.1, 0.1, ...]}"
```

### 6. Reinforcement Learning
*   **Training:** `python3 train_rl.py`
*   **Evaluation:** `python3 eval_rl.py`

Logs and models are saved in the `logs/` directory.

---
**Note:** If you encounter `ModuleNotFoundError: No module named 'rclpy'`, ensure you have sourced `/opt/ros/humble/setup.bash` **before** activating your virtual environment.
