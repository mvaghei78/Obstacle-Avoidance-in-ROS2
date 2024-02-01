#!/bin/bash

# Your commands
colcon build --packages-select obstacle_avoidance
source install/setup.bash
ros2 launch obstacle_avoidance obstacle_avoidance.launch.py
