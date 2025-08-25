# Overview

Hand Controller making use of MediaPipe models and ASL recognition.


The mediapipe models are inferenced using the following python implementation:
   - Blaze Utility (python version) : [blaze_app_python](https://github.com/albertabeef/blaze_app_python)

For a version using Google MediaPipe, please refer to the following alternate implementation:
   - ASL MediaPipe PointNet : [asl_mediapipe_pointnet](https://github.com/albertabeef/asl_mediapipe_pointnet)



## Cloning the repo

The repo must be cloned with the --recursive option to clone the blaze_app_python submodule:

   - git clone --recursive https://github.com/AlbertaBeef/hand_controller
   - cd hand_controller

If already cloned without the --recursive option, the blaze_app_python content can be obtained as follows:

   - cd hand_controller
   - git submodule update --recursive


## Downloading the ASL model

The ASL pointnet (PyTorch) model can be downloaded as follows:

   - cd asl_pointnet
   - source ./get_model.sh
   - cd ..


## Downloading the MediaPipe models

The original TFLite MediaPipe models can be downloaded as follows:

   - cd blaze_app_python/blaze_tflite/models
   - source ./get_tflite_models.sh
   - cd ../../..

The converted PyTorch MediaPipe models can be downloaded as follows:

   - cd blaze_app_python/blaze_pytorch/models
   - source ./get_pytorch_models.sh
   - cd ../../..

The Hailo-8 optimized MediaPipe models can be downloaded as follows:

   - cd blaze_app_python/blaze_hailo/models
   - source ./get_hailo8_models.sh
   - unzip blaze_hailo8_models.zip
   - cp hailo8/*.hef .
   - cd ../../..



## Installing Dependencies

The following dependencies are required:

   - tensorflow
   - pytorch
   - opencv


## Stand-Alone Operation

Three variations of a hand controller with mediapipe models and ASL are provided:
   - hand_controller_tflite_asl.py
   - hand_controller_pytorch_asl.py
   - hand_controller_hailo8_asl.py

To launch the ASL hand controller, using the TFLite version of blaze_app_python, invoke as follows:
   - python3 hand_controller_tflite_asl.py

![](images/hand_controller_tflite_asl.gif)
   
One variation of a hand controller with mediapipe models and visual dials is provided:
   - hand_controller_tflite_dials.py

To launch the "dials" hand controller, using the TFLite version of blaze_app_python, invoke as follows:
   - python3 hand_controller_tflite_dials.py

![](images/hand_controller_tflite_dials.gif)



## Use as ROS2 Node

Itialize the ROS2 environment:

   - /opt/ros2/humble/setup.bash

Build and Install the ROS2 packages:

   - cd ros2_ws/hand_controller
   - colcon build
   - source install/setup.bash


The hand_controller package provides three controllers:

   - hand_controller_asl_twist_node : used to control a vehicle (turtlesim, MOGI-ROS vehicle, ROSMASTER-X3)
      - recognized hand signs (A:advance, B:back-up, L:turn-left, R:turn-right)
      - generates Twist messages

   - hand_controller_asl_joints_node : used to control the MOGI-ROS robotic arm
      - recognized left hand signs (A:advance, B:back-up, L:left, R:right, U:up, D:down)
      - recognized right hand signs (A:close-gripper, B:open-gripper)
      - generates JointTrajectory messages

   - hand_controller_asl_pose_node : used to control the MYCOBOT-280 robotic arm
      - recognized hand signs (A:advance, B:back-up, L:left, R:right, U:up, D:down)
      - reads current pose of robotic arm (position of gripper wrt base)
      - generates Pose messages for current position and target position
      - communicates with MoveIt2 to plan/execute robotic arm movement

For convenience, the package also contains the following nodes:

   - usbcam_publisher.py
      - searches for /dev/video* & /dev/media* corresponding to USB camera
      - opens USB camera in 640x480 resolution
      - published images as "image_raw" topic

   - usbcam_subscriber.py
      - subscribes to "image_raw" topic
      - displays images with cv2.imshow


All of the demos are partitionned in 2 parts:
   - Part 1 can be executed on embedded hardware (without any ability to display)
   - Part 2 is meant to be executed on a host PC with Gazebo simulation.



Launch the hand_controller_asl_twist node with usbcam_publisher and turtlesim nodes:

   - ros2 launch hand_controller demo01_turtlesim_part1_asl.launch.py | ros2 launch hand_controller demo01_turtlesim_part2.launch.py

![](https://github.com/AlbertaBeef/asl_mediapipe_pointnet/blob/main/images/asl_mediapipe_pointnet_demo01_ros2_turtlesim.gif)



## Use as ROS2 Node to control Vehicules in Gazebo simulator

Launch the hand_controller_asl_twist node with MOGI-ROS vehicle:

   - ros2 launch hand_controller demo11_mogiros_car_part1_asl.launch.py | ros2 launch hand_controller demo11_mogiros_car_part2.launch.py

Control Vehicle with Hand Signs

   - A : Advance
   - B : Backup
   - L : Turn Left
   - R : Turn Right

![](https://github.com/AlbertaBeef/asl_mediapipe_pointnet/blob/main/images/asl_mediapipe_pointnet_demo01_ros2_gazebo.gif)

Launch the hand_controller_asl_twist node with ROSMASTER-X3 vehicle:

   - ros2 launch hand_controller demo12_rosmaster_part1_asl.launch.py | ros2 launch hand_controller demo12_rosmaster_part2.launch.py

Control Vehicle with Hand Signs

   - A : Advance
   - B : Backup
   - L : Turn Left
   - R : Turn Right

![](https://github.com/AlbertaBeef/asl_mediapipe_pointnet/blob/main/images/asl_mediapipe_pointnet_demo02_ros2_gazebo_rosmaster.gif)


## Use as ROS2 Node to control Robotic Arms in Gazebo simulator

Launch the hand_controller_asl_pose node with MOGI-ROS simple robotic arm:

   - ros2 launch hand_controller demo21_mogiros_arm_part1_asl.launch.py | ros2 launch hand_controller demo21_mogiros_arm_part2.launch.py

Control Robotic Arm with Left/Right Hands:

   - Left Hand
      - L : Turn Arm Left
      - R : Turn Arm Right
      - A : Advance Arm (shoulder joint)
      - B : Backup Arm (shoulder joint)
      - U : Lift Arm (elbow joint)
      - Y : Lower Arm (elbow joint)

   - Right Hand
      - A : Close Gripper
      - B : Open Gripper

![](https://github.com/AlbertaBeef/asl_mediapipe_pointnet/blob/main/images/asl_mediapipe_pointnet_demo04_ros2_gazebo_mogiros_arm.gif)

Launch the hand_controller_asl_pose node with MYCOBOT-280 robotic arm:

   - moveit &
   - ros2 launch hand_controller demo31_mycobot_part1_asl.launch.py | ros2 launch hand_controller demo31_mycobot_part2.launch.py


Control Robotic Arm with Hand Signs

   - L : Move Left
   - R : Move Right
   - A : Move Forward
   - B : Move Backward
   - U : Move Up
   - Y : Move Down

![](https://github.com/AlbertaBeef/asl_mediapipe_pointnet/blob/main/images/asl_mediapipe_pointnet_demo03_ros2_gazebo_mycobot.gif)


## References

The Complete Guide to Docker for ROS 2 Jazzy Projects
   - https://automaticaddison.com/the-complete-guide-to-docker-for-ros-2-jazzy-projects/

Automatic Addison on-line Tutorials:
   - https://automaticaddison.com/tutorials
   - https://github.com/automaticaddison/mycobot_ros2 (branch=jazzy)
   - https://github.com/automaticaddison/yahboom_rosmaster

MOGI-ROS on-line Tutorials:
   - https://github.com/MOGI-ROS/Week-1-2-Introduction-to-ROS2
   - https://github.com/MOGI-ROS/Week-3-4-Gazebo-basics
   - https://github.com/MOGI-ROS/Week-5-6-Gazebo-sensors
   - https://github.com/MOGI-ROS/Week-7-8-ROS2-Navigation
   - https://github.com/MOGI-ROS/Week-9-10-Simple-arm

Accelerating MediaPipe:
   - Hackster Series Part 1 [Blazing Fast Models](https://avnet.me/mediapipe-01-models)
   - Hackster Series Part 2 [Insightful Datasets for ASL recognition](https://avnet.me/mediapipe-02-datasets)
   - Hackster Series Part 3 [Accelerating the MediaPipe models with Vitis-AI 3.5](https://avnet.me/mediapipe-03-vitis-ai-3.5)
   - Hackster Series Part 4 [Accelerating the MediaPipe models with Hailo-8](https://avnet.me/mediapipe-04-Hailo-8)
   - Hackster Series Part 5 [Accelerating the MediaPipe models on RPI5 AI Kit](https://avnet.me/mediapipe-05-rpi5aikit)
   - Hackster Series Part 6 [Accelerating the MediaPipe models with MemryX](https://avnet.me/mediapipe-06-memryx)
   - Blaze Utility (python version) : [blaze_app_python](https://github.com/albertabeef/blaze_app_python)
   - Blaze Utility (C++ version) : [blaze_app_cpp](https://github.com/albertabeef/blaze_app_cpp)

ASL Recognition using PointNet (by Edward Roe):
   - Medium Article [ASL Recognition using PointNet and MediaPipe](https://medium.com/@er_95882/asl-recognition-using-pointnet-and-mediapipe-f2efda78d089)
   - Kaggle Dataset [American Sign Language Dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset)
   - GitHub Source [pointnet_hands](https://github.com/e-roe/pointnet_hands/tree/main)
