#include "pathfinder_node.hpp"

PathFinderNode::PathFinderNode() : Node("pathfinder_node")
{
  publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("output_topic", 10);

  subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "input_topic", 10, std::bind(&PathFinderNode::topic_callback, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(), "PathFinder Node started");
}

void PathFinderNode::topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  std::string log_msg = "Received array: [";
  for (size_t i = 0; i < msg->data.size(); ++i)
  {
    log_msg += std::to_string(msg->data[i]);
    if (i < msg->data.size() - 1) log_msg += ", ";
  }
  log_msg += "]";

  RCLCPP_INFO(this->get_logger(), "%s", log_msg.c_str());

  auto out_msg = std_msgs::msg::Float64MultiArray();
  out_msg.data = msg->data;

  publisher_->publish(out_msg);
}

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathFinderNode>());
  rclcpp::shutdown();
  return 0;
}
