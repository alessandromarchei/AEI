import rclpy
from rclpy.node import Node

import carla
import math
import numpy as np
from builtin_interfaces.msg import Time 
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

LOOKAHEAD_DISTANCE = 60.0  # meters
STEP_DISTANCE = 2.0        # distance between waypoints
LANE_WIDTH = 3.5          # meters, typical lane width

def yaw_to_quaternion(yaw_deg):
    yaw = math.radians(yaw_deg)
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(yaw / 2.0),
        "w": math.cos(yaw / 2.0)
    }
def rpy_to_matrix(roll, pitch, yaw):
    """Return 3x3 rotation matrix from roll, pitch, yaw (in radians)"""
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,            cp*cr]
    ])
    return R

class RoadShapePublisher(Node):
    def __init__(self):
        super().__init__('road_shape_publisher')
        
        self.egoPath_viz_pub_ = self.create_publisher(Path, '/viz/egoPath', 2)
        self.egoLaneL_viz_pub_ = self.create_publisher(Path, '/viz/egoLaneL', 2)
        self.egoLaneR_viz_pub_ = self.create_publisher(Path, '/viz/egoLaneR', 2)
        
        self.egoLaneL_pub_ = self.create_publisher(Float32MultiArray, '/egoLaneL', 2)
        self.egoLaneR_pub_ = self.create_publisher(Float32MultiArray, '/egoLaneR', 2)
        self.egoPath_pub_ = self.create_publisher(Float32MultiArray, '/egoPath', 2)
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.ego = self._find_ego_vehicle()
        if self.ego is None:
            self.get_logger().error('Ego vehicle not found, exiting.')
            rclpy.shutdown()
            return

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.egopath_pts = []
        self.egolaneL_pts = []
        self.egolaneR_pts = []

    def _find_ego_vehicle(self):
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.attributes.get('role_name') == 'hero':
                return actor
        self.get_logger().error('Ego vehicle not found')
        return None
    
    def publish_array(self, pub, pts: np.ndarray):
        """
        Publish Nx3 numpy array as Float32MultiArray
        """
        msg = Float32MultiArray()
        msg.data = pts.astype(np.float32).flatten().tolist()

        # layout metadata
        msg.layout.dim.append(MultiArrayDimension(label="points",
                                                  size=pts.shape[0],
                                                  stride=pts.shape[0] * pts.shape[1]))
        msg.layout.dim.append(MultiArrayDimension(label="xy",
                                                  size=pts.shape[1],
                                                  stride=pts.shape[1]))

        pub.publish(msg)
        
    def timer_callback(self):
        if not self.ego:
            return

        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        ego_rot = ego_tf.rotation
        ego_yaw = math.radians(ego_rot.yaw)
        ego_pitch = -math.radians(ego_rot.pitch) # CARLA uses left-handed coordinate system
        ego_roll = math.radians(ego_rot.roll)

        R_world_to_ego = rpy_to_matrix(ego_roll, ego_pitch, ego_yaw).T  # inverse = transpose

        snapshot = self.world.get_snapshot()
        elapsed = snapshot.timestamp.elapsed_seconds

        # Create ROS time
        ros_time = Time()
        ros_time.sec = int(elapsed)
        ros_time.nanosec = int((elapsed - ros_time.sec) * 1e9)

        curr_wp = self.map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        total_dist = 0.0
        
        path_msg = Path()
        path_msg.header.stamp = ros_time
        path_msg.header.frame_id = "hero"
        path_msg.poses = []

        left_lane = Path()
        right_lane = Path()
        left_lane.header = path_msg.header
        right_lane.header = path_msg.header

        while total_dist < LOOKAHEAD_DISTANCE and curr_wp is not None:
            wp_loc = curr_wp.transform.location
            wp_pos = np.array([wp_loc.x - ego_loc.x,
                            wp_loc.y - ego_loc.y,
                            wp_loc.z - ego_loc.z])
            local_pos = R_world_to_ego @ wp_pos  # rotate to ego frame

            ps = PoseStamped()
            ps.header.stamp = ros_time
            ps.header.frame_id = "hero"
            ps.pose.position.x = local_pos[0]
            ps.pose.position.y = -local_pos[1] # CARLA uses left-handed coordinate system
            ps.pose.position.z = local_pos[2]

            wp_yaw = math.radians(curr_wp.transform.rotation.yaw)
            relative_yaw = wp_yaw - ego_yaw
            q = yaw_to_quaternion(math.degrees(relative_yaw))
            ps.pose.orientation.x = q["x"]
            ps.pose.orientation.y = q["y"]
            ps.pose.orientation.z = q["z"]
            ps.pose.orientation.w = q["w"]

            path_msg.poses.append(ps)

            # Advance
            next_wps = curr_wp.next(STEP_DISTANCE)
            if not next_wps:
                break
            next_wp = next_wps[0]
            total_dist += curr_wp.transform.location.distance(next_wp.transform.location)
            curr_wp = next_wp

            # Create left lane
            left_ps = PoseStamped()
            left_ps.header = ps.header
            left_ps.pose.position.x = ps.pose.position.x - LANE_WIDTH / 2 * math.sin(-relative_yaw)
            left_ps.pose.position.y = ps.pose.position.y + LANE_WIDTH / 2 * math.cos(-relative_yaw)
            left_ps.pose.position.z = ps.pose.position.z
            left_ps.pose.orientation = ps.pose.orientation
            left_lane.poses.append(left_ps)

            # Create right lane
            right_ps = PoseStamped()
            right_ps.header = ps.header
            right_ps.pose.position.x = ps.pose.position.x + LANE_WIDTH / 2 * math.sin(-relative_yaw)
            right_ps.pose.position.y = ps.pose.position.y - LANE_WIDTH / 2 * math.cos(-relative_yaw)
            right_ps.pose.position.z = ps.pose.position.z
            right_ps.pose.orientation = ps.pose.orientation
            right_lane.poses.append(right_ps)
            
        if path_msg.poses:
            self.get_logger().info(f'Publishing path with {len(path_msg.poses)} waypoints')
            self.egoPath_viz_pub_.publish(path_msg)
            self.egoLaneL_viz_pub_.publish(left_lane)
            self.egoLaneR_viz_pub_.publish(right_lane)  
            egopath_pts  = np.array([[pose.pose.position.x, pose.pose.position.y] for pose in path_msg.poses])
            egolaneL_pts = np.array([[pose.pose.position.x, pose.pose.position.y] for pose in left_lane.poses])
            egolaneR_pts = np.array([[pose.pose.position.x, pose.pose.position.y] for pose in right_lane.poses])
            self.publish_array(self.egoPath_pub_,  egopath_pts)
            self.publish_array(self.egoLaneR_pub_, egolaneR_pts)
            self.publish_array(self.egoLaneL_pub_, egolaneL_pts)
            self.get_logger().info("Published egoPath, egoLaneL, egoLaneR")

        
def main(args=None):
    rclpy.init(args=args)
    node = RoadShapePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
