#!/bin/bash
# run_sim.sh

echo "Sourcing ROS Jazzy..."
source /opt/ros/jazzy/setup.bash

echo "Sourcing Local Workspace..."
if [ -f "install/setup.bash" ]; then
    source install/setup.bash
else
    echo "Warning: install/setup.bash not found. Building first..."
fi

echo "Building Package..."
colcon build --symlink-install

echo "Sourcing Local Workspace (post-build)..."
source install/setup.bash

echo "Launching Robot..."
ros2 launch takos_bringup sim.launch.py
