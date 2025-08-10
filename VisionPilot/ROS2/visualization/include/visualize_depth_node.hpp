#ifndef VISUALIZE_DEPTH_NODE_HPP_
#define VISUALIZE_DEPTH_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

namespace autoware_pov::visualization
{

class VisualizeDepthNode : public rclcpp::Node
{
public:
  explicit VisualizeDepthNode(const rclcpp::NodeOptions & options);

private:
  void onData(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

  // ROS
  image_transport::Subscriber sub_;
  image_transport::Publisher pub_;
};

}  // namespace autoware_pov::visualization

#endif  // VISUALIZE_DEPTH_NODE_HPP_