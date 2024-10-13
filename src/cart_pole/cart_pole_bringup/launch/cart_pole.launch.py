import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    gazebo = IncludeLaunchDescription(
        os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py"),
        launch_arguments=[("gz_args", ["-r -v 4 empty.sdf"])],
    )

    gazebo_spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_cart_pole",
        arguments=["-name", "spawn_cart_pole", "-topic", "robot_description"],
        output="screen",
    )

    xacro_file = os.path.join(get_package_share_directory('cart_pole_bringup'), 'urdf', 'cart_pole.urdf.xacro')
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[
            {'robot_description': Command(['xacro ', xacro_file]) },
            {'use_sim_time' : True}

        ],
    )

    joint_state_broadcaster_spawner = ExecuteProcess(
        name="joint_state_broadcaster_spawner",
        cmd=["ros2", "control", "load_controller", "--set-state", "active", "joint_state_broadcaster"],
        shell=True,
        output="screen",
    )

    effort_controller_spawner = ExecuteProcess(
        name="velocity_controller_spawner",
        cmd=[ "ros2", "control", "load_controller", "--set-state", "active", "effort_controller"],
        shell=True,
        output="screen",
    )

    return LaunchDescription([
        gazebo,
        gazebo_spawn_robot,
        robot_state_publisher,
        joint_state_broadcaster_spawner,
        effort_controller_spawner
    ])