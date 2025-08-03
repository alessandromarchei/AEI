#include "pathfinder_node.hpp"

PathFinderNode::PathFinderNode() : Node("pathfinder_node")
{
  bayesFilter = Estimator();
  bayesFilter.configureFusionGroups({
      // {start_idx,end_idx}
      // fuse indices 1,2 → result in index 3
      {0, 3}, // cte1, cte2 → fused at index 3
      {5, 7}, // yaw1, yaw2 → fused at index 7
      {9, 11} // curv1, curv2 → fused at index 11
  });
  Gaussian default_state = {0.0, 1e6}; // states can be any value, variance is large
  std::array<Gaussian, STATE_DIM> init_state;
  init_state.fill(default_state);
  bayesFilter.initialize(init_state);

  publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("output_topic", 10);

  subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "input_topic", 10, std::bind(&PathFinderNode::topic_callback, this, std::placeholders::_1));

  timer_ = this->create_wall_timer(std::chrono::milliseconds(1000), std::bind(&PathFinderNode::timer_callback_, this)
);

  RCLCPP_INFO(this->get_logger(), "PathFinder Node started");
}

void PathFinderNode::timer_callback_()
{

  Gaussian meas = {0.0, 0.5};
  Gaussian proc = {0.0, 0.0};
  std::array<Gaussian, STATE_DIM> measurement;
  std::array<Gaussian, STATE_DIM> process;
  measurement.fill(meas);
  process.fill(proc);

  bayesFilter.update(measurement);
  const auto &state = bayesFilter.getState();
  bayesFilter.predict(process);

  std::string mean_log_msg = "Mean: [";
  std::string var_log_msg = "Var:  [";
  for (auto s:state) {
    mean_log_msg += std::to_string(s.mean) + "  ";
    var_log_msg += std::to_string(s.variance) + "  ";
  }
  mean_log_msg += "]";
  var_log_msg += "]";
  RCLCPP_INFO(this->get_logger(), mean_log_msg.c_str());
  RCLCPP_INFO(this->get_logger(), var_log_msg.c_str());

  auto out_msg = std_msgs::msg::Float64MultiArray();
  publisher_->publish(out_msg);
}

void PathFinderNode::topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  std::array<Gaussian, STATE_DIM> measurement;
  std::array<Gaussian, STATE_DIM> process;
  bayesFilter.update(measurement);
  const auto &state = bayesFilter.getState();
  bayesFilter.predict(process);

  std::string log_msg = "Received array: [";
  for (size_t i = 0; i < msg->data.size(); ++i)
  {
    log_msg += std::to_string(msg->data[i]);
    if (i < msg->data.size() - 1)
      log_msg += ", ";
  }
  log_msg += "]";

  RCLCPP_INFO(this->get_logger(), "%s", log_msg.c_str());

  auto out_msg = std_msgs::msg::Float64MultiArray();
  out_msg.data = msg->data;

  publisher_->publish(out_msg);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathFinderNode>());
  rclcpp::shutdown();
  return 0;
}
