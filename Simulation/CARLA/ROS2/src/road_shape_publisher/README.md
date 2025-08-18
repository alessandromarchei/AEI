## Road Shape Publisher
Simulates EgoLanes and EgoPath using ground truth data from CARLA Python API, publishes 
-  ground truth points in BEV metric coordinate relative to vehicle `std_msgs/Float32MultiArray`
-  same data as above in `nav_msgs/Path` for visualization