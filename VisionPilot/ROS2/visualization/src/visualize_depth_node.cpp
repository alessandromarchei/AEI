#include "visualize_depth_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <rclcpp_components/register_node_macro.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace autoware_pov::visualization
{

VisualizeDepthNode::VisualizeDepthNode(const rclcpp::NodeOptions & options)
: Node("visualize_depth_node", options)
{
  // Parameters
  const std::string depth_topic = this->declare_parameter<std::string>("depth_topic");
  const std::string output_topic = this->declare_parameter<std::string>("output_topic", "~/out/image");

  // Publisher
  pub_ = image_transport::create_publisher(this, output_topic);

  // Subscriber
  sub_ = image_transport::create_subscription(
    this, depth_topic, std::bind(&VisualizeDepthNode::onData, this, std::placeholders::_1), "raw");
}

void VisualizeDepthNode::onData(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
  cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);

  // Normalize depth to 0-255 range
  cv::Mat normalized_depth;
  double min_val, max_val;
  cv::minMaxLoc(depth_ptr->image, &min_val, &max_val);
  
  if (max_val > min_val) {
    depth_ptr->image.convertTo(normalized_depth, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
  } else {
    normalized_depth = cv::Mat::zeros(depth_ptr->image.size(), CV_8UC1);
  }
  
  // Apply colormap
  cv::Mat colorized_depth;
  cv::applyColorMap(normalized_depth, colorized_depth, cv::COLORMAP_VIRIDIS);

  sensor_msgs::msg::Image::SharedPtr out_msg =
    cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, colorized_depth).toImageMsg();
  pub_.publish(out_msg);
}

}  // namespace autoware_pov::visualization

RCLCPP_COMPONENTS_REGISTER_NODE(autoware_pov::visualization::VisualizeDepthNode)