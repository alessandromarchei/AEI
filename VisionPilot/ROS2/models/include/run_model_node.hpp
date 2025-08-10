#ifndef RUN_MODEL_NODE_HPP_
#define RUN_MODEL_NODE_HPP_

#include "inference_backend_base.hpp"
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <memory>

namespace autoware_pov::vision
{

class RunModelNode : public rclcpp::Node
{
public:
  explicit RunModelNode(const rclcpp::NodeOptions & options);

private:
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  // Parameters
  std::string model_type_;
  std::string output_topic_str_;
  
  // Backend
  std::unique_ptr<InferenceBackend> backend_;

  // ROS
  image_transport::Subscriber sub_;
  image_transport::Publisher pub_;
};

}  // namespace autoware_pov::vision

#endif  // RUN_MODEL_NODE_HPP_