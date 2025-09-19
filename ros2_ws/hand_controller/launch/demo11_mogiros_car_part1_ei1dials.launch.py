from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "repo_path",
            default_value="/root/hand_controller",
            description="Path (absolute) to hand_controller repo."
        ),
        DeclareLaunchArgument(
            "model",
            default_value="/root/hand_controller/handsv2.eim",
            description="Name of Edge Impulse FOMO model."
        ),       
        DeclareLaunchArgument(
            "verbose",
            default_value="True",
            description="Verbose mode."
        ),               
        DeclareLaunchArgument(
            "use_imshow",
            default_value="False",
            description="Enable OpenCV display."
        ),
        Node(
            package='hand_controller',
            executable='usbcam_publisher_node',
            name="usbcam_publisher",
            remappings=[("image_raw", "usbcam_image")]            
        ),
        Node(
            package='hand_controller',
            executable='hand_controller_ei1dials_twist_node',
            name="hand_controller",
            parameters=[
               {"repo_path":LaunchConfiguration("repo_path")},
               {"model":LaunchConfiguration("model")},
               {"verbose":PythonExpression(['"', LaunchConfiguration('verbose'), '" == "True"'])},
               {"use_imshow":PythonExpression(['"', LaunchConfiguration('use_imshow'), '" == "True"'])}
            ],
            remappings=[
               ("image_raw", "usbcam_image"),
               ("hand_controller/cmd_vel", "cmd_vel")
            ]            
        )
    ])
