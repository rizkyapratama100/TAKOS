import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Find the ros_gz_sim package
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # 2. Find the world file and models
    pkg_takos_bringup = get_package_share_directory('takos_bringup')
    world_file = os.path.join(pkg_takos_bringup, 'resource', 'tentacle.sdf')
    models_path = os.path.join(pkg_takos_bringup, 'resource', 'models')

    # 3. Define the Gazebo launch file we want to include
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': [world_file, ' -r']}.items(),
    )

    # 4. Bridge ROS 2 and Gazebo Sim
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
        ],
        output='screen'
    )

    # 5. Set the GZ_SIM_RESOURCE_PATH
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[models_path]
    )

    # 6. Create the Launch Description
    return LaunchDescription([
        set_gz_resource_path,
        gazebo,
        bridge
    ])