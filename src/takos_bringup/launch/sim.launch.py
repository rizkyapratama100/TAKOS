import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # 1. Find the ros_gz_sim package
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # 2. Define the Gazebo launch file we want to include
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': 'empty.sdf -r'}.items(),
    )

    # 3. Create the Launch Description
    return LaunchDescription([
        gazebo
    ])