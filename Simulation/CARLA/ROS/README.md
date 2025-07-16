# Configuring CARLA

Spawning Ego vehicle

Sensors, enable_for_ros()

Spectator view

Waiting for additional towns with highways to be available in CARLA 0.10


## CARLA-Autoware Custom Interfaces
- waypoints_publisher : creates `autoware_planning_msgs/Trajectory`
- control_msg_converter: converts autoware controller output from `autoware_control_msgs/Control` to `ros_carla_msgs/CarlaEgoVehicleControl.msg`