#ifndef VISUALIZE_MASKS_NODE_HPP_
#define VISUALIZE_MASKS_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace autoware_pov::visualization
{

class VisualizeMasksNode : public rclcpp::Node
{
public:
  explicit VisualizeMasksNode(const rclcpp::NodeOptions & options);

private:
  using Image = sensor_msgs::msg::Image;
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  void onData(const Image::ConstSharedPtr& image, const Image::ConstSharedPtr& mask);
  void createColorMap(const std::string& viz_type);

  // Parameters
  std::vector<cv::Vec3b> color_map_;
  std::string viz_type_;

  // ROS
  image_transport::Publisher pub_;
  message_filters::Subscriber<Image> sub_image_;
  message_filters::Subscriber<Image> sub_mask_;
  std::shared_ptr<Synchronizer> sync_;
};

}  // namespace autoware_pov::visualization

#endif  // VISUALIZE_MASKS_NODE_HPP_