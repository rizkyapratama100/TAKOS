FROM osrf/ros:jazzy-desktop

# 1. Install ROS/Gazebo tools
# We removed 'python3-black' from here to avoid the "Unable to locate package" error
RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-xacro \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    python3-pip \
    nano \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. THE FIX: Install Python tools safely
# --break-system-packages: Allows installation on Ubuntu 24.04
# --ignore-installed: Prevents the "Cannot uninstall filelock" error by overwriting dependencies
RUN pip3 install --break-system-packages --ignore-installed black pre-commit

# 3. Setup the environment
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc