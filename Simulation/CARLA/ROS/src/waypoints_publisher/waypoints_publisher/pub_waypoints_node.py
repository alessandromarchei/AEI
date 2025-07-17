import rclpy
from rclpy.node import Node

from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Duration

import carla
import math

import matplotlib.pyplot as plt

LOOKAHEAD_DISTANCE = 100.0  # meters
STEP_DISTANCE = 2.0        # distance between waypoints
DEFAULT_SPEED = 10.0       # m/s constant assumed speed

def yaw_to_quaternion(yaw_deg):
    yaw = math.radians(yaw_deg)
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(yaw / 2.0),
        "w": math.cos(yaw / 2.0)
    }

class CarlaTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('carla_trajectory_publisher')
        self.publisher_ = self.create_publisher(Trajectory, '/planning/trajectory', 10)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.ego = self._find_ego_vehicle()
        
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(10, 8)
        self.line, = self.ax.plot([], [], 'b.-', label="Current Trajectory")
        self.ax.set_title("Ego Trajectory")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_aspect('equal', adjustable='box')  # <- preserves x/y scale while fitting limits
        self.ax.legend()
        limit = 200
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.grid(True)
        plt.ion()
        plt.show()

    def _find_ego_vehicle(self):
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.attributes.get('role_name') == 'hero':
                return actor
        self.get_logger().error('Ego vehicle not found')
        return None

    def timer_callback(self):
        if not self.ego:
            return

        ego_transform = self.ego.get_transform()
        location = ego_transform.location
        waypoint = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

        if waypoint is None:
            self.get_logger().warn("No valid waypoint found.")
            return

        traj_msg = Trajectory()
        total_dist = 0.0
        curr_wp = waypoint
        time_from_start = 0.0
        traj_points = []
        plot_x = []
        plot_y = []

        while total_dist < LOOKAHEAD_DISTANCE and curr_wp is not None:
            pt = TrajectoryPoint()

            # Pose
            pt.pose = Pose()
            pt.pose.position.x = curr_wp.transform.location.x
            pt.pose.position.y = curr_wp.transform.location.y
            pt.pose.position.z = curr_wp.transform.location.z

            q = yaw_to_quaternion(curr_wp.transform.rotation.yaw)
            pt.pose.orientation.x = q["x"]
            pt.pose.orientation.y = q["y"]
            pt.pose.orientation.z = q["z"]
            pt.pose.orientation.w = q["w"]

            # Kinematic fields
            pt.longitudinal_velocity_mps = DEFAULT_SPEED
            pt.lateral_velocity_mps = 0.0
            pt.acceleration_mps2 = 0.0
            pt.heading_rate_rps = 0.0
            pt.front_wheel_angle_rad = 0.0
            pt.rear_wheel_angle_rad = 0.0

            # Time
            time_msg = Duration()
            time_msg.sec = int(time_from_start)
            time_msg.nanosec = int((time_from_start - int(time_from_start)) * 1e9)
            pt.time_from_start = time_msg

            traj_points.append(pt)

            # For plotting
            plot_x.append(-pt.pose.position.x)
            plot_y.append(pt.pose.position.y)

            # Advance to next waypoint
            next_wps = curr_wp.next(STEP_DISTANCE)
            if not next_wps:
                break
            next_wp = next_wps[0]
            dist = curr_wp.transform.location.distance(next_wp.transform.location)
            total_dist += dist
            time_from_start += dist / DEFAULT_SPEED
            curr_wp = next_wp

        traj_msg.points = traj_points
        self.publisher_.publish(traj_msg)
        self.get_logger().info(f'Published {len(traj_points)} trajectory points.')

        # Update live plot
        self.line.set_data(plot_x, plot_y)
        self.ax.plot(-location.x, location.y, 'r.', label="Ego Vehicle")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    node = CarlaTrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
