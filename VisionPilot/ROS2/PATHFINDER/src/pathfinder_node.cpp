#include "pathfinder_node.hpp"

PathFinderNode::PathFinderNode() : Node("pathfinder_node"), drivCorr(std::nullopt, std::nullopt, std::nullopt)
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

  publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("tracked_states", 10);

  sub_laneL_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/egoLaneL", 10,
                                                                           std::bind(&PathFinderNode::callbackLaneL, this, std::placeholders::_1));
  sub_laneR_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/egoLaneR", 10,
                                                                           std::bind(&PathFinderNode::callbackLaneR, this, std::placeholders::_1));
  sub_path_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/egoPath", 10,
                                                                          std::bind(&PathFinderNode::callbackPath, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(), "PathFinder Node started");
}

void PathFinderNode::callbackLaneL(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  auto points = reshapeTo2D(msg);
  RCLCPP_INFO(this->get_logger(), "Received egoLaneL with %zu points", points.size());
}

void PathFinderNode::callbackLaneR(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  auto points = reshapeTo2D(msg);
  RCLCPP_INFO(this->get_logger(), "Received egoLaneR with %zu points", points.size());
}

void PathFinderNode::callbackPath(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  auto points = reshapeTo2D(msg);
  RCLCPP_INFO(this->get_logger(), "Received egoPath with %zu points", points.size());
}

/// Convert Flat Float32MultiArray → vector of {x,y}
std::vector<std::array<float, 2>> PathFinderNode::reshapeTo2D(const std_msgs::msg::Float32MultiArray::SharedPtr &msg)
{
  size_t N = 0;
  size_t D = 0;
  if (msg->layout.dim.size() >= 2)
  {
    N = msg->layout.dim[0].size; // number of points
    D = msg->layout.dim[1].size; // should be 2
  }

  std::vector<std::array<float, 2>> pts;
  if (D != 2 || N == 0)
  {
    RCLCPP_WARN(this->get_logger(), "Invalid dimensions (N=%zu, D=%zu)", N, D);
    return pts;
  }

  pts.reserve(N);
  for (size_t i = 0; i < N; i++)
  {
    std::array<float, 2> p;
    p[0] = msg->data[i * D + 0]; // x
    p[1] = msg->data[i * D + 1]; // y
    pts.push_back(p);
  }
  return pts;
}

//TODO: rewrite the following function to use the new callback functions
void PathFinderNode::topic_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  std::vector<cv::Point2f> list1;
  std::vector<cv::Point2f> list2;

  // Check if layout has correct dimensions
  if (msg->layout.dim.size() != 3)
  {
    RCLCPP_ERROR(rclcpp::get_logger("unpack"), "Invalid layout: expected 3 dimensions");
    return;
  }

  size_t num_lists = msg->layout.dim[0].size;  // Should be 2
  size_t max_points = msg->layout.dim[1].size; // max points per list
  size_t xy_size = msg->layout.dim[2].size;    // Should be 2

  if (num_lists != 2 || xy_size != 2)
  {
    RCLCPP_ERROR(rclcpp::get_logger("unpack"), "Unexpected layout sizes");
    return;
  }

  const auto &data = msg->data;

  if (data.size() < num_lists * max_points * xy_size)
  {
    RCLCPP_ERROR(rclcpp::get_logger("unpack"), "Data size does not match layout");
    return;
  }

  // Reserve memory
  list1.reserve(max_points);
  list2.reserve(max_points);

  // Parse first list
  size_t offset_list1 = 0;
  size_t offset_list2 = max_points * xy_size; // second list starts after first list block

  for (size_t i = 0; i < max_points; ++i)
  {
    float x1 = data[offset_list1 + i * xy_size];
    float y1 = data[offset_list1 + i * xy_size + 1];

    if (!std::isnan(x1) && !std::isnan(y1))
    {
      list1.emplace_back(x1, y1);
    }

    float x2 = data[offset_list2 + i * xy_size];
    float y2 = data[offset_list2 + i * xy_size + 1];

    if (!std::isnan(x2) && !std::isnan(y2))
    {
      list2.emplace_back(x2, y2);
    }
  }
  auto coeff1 = fitQuadPoly(list1);
  auto coeff2 = fitQuadPoly(list2);

  drivCorr = drivingCorridor(fittedCurve(coeff1), fittedCurve(coeff2), std::nullopt);

  std::array<Gaussian, STATE_DIM> measurement;
  std::array<Gaussian, STATE_DIM> process;

  std::random_device rd;
  std::default_random_engine generator(rd());
  std::uniform_real_distribution<double> dist(-epsilon, epsilon);
  for (size_t i = 0; i < STATE_DIM; ++i)
  {
    process[i].mean = dist(generator);
    process[i].variance = proc_SD * proc_SD;
    measurement[i].variance = meas_SD * meas_SD;
  }

  const auto &egoPath = drivCorr.egoPath;
  const auto &egoLaneL = drivCorr.egoLaneL;
  const auto &egoLaneR = drivCorr.egoLaneR;

  measurement[0].mean = egoPath->cte;
  measurement[1].mean = egoLaneL->cte - drivCorr.width / 2.0;
  measurement[2].mean = egoLaneR->cte + drivCorr.width / 2.0;
  measurement[3].mean = 0.0;

  measurement[4].mean = egoPath->yaw_error;
  measurement[5].mean = egoLaneL->yaw_error;
  measurement[6].mean = egoLaneR->yaw_error;
  measurement[7].mean = 0.0;

  measurement[8].mean = egoPath->curvature;
  measurement[9].mean = egoLaneL->curvature;
  measurement[10].mean = egoLaneR->curvature;
  measurement[11].mean = 0.0;

  measurement[12].mean = drivCorr.width;

  bayesFilter.update(measurement);
  const auto &state = bayesFilter.getState();
  bayesFilter.predict(process);

  std::string mean_log_msg = "Mean: [";
  std::string var_log_msg = "Var:  [";
  for (auto s : state)
  {
    mean_log_msg += std::to_string(s.mean) + "  ";
    var_log_msg += std::to_string(s.variance) + "  ";
  }
  mean_log_msg += "]";
  var_log_msg += "]";
  RCLCPP_INFO(this->get_logger(), mean_log_msg.c_str());
  RCLCPP_INFO(this->get_logger(), var_log_msg.c_str());

  auto out_msg = std_msgs::msg::Float32MultiArray();
  out_msg.data.resize(STATE_DIM * 2);
  out_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
  out_msg.layout.dim[0].label = "mean";
  out_msg.layout.dim[0].size = STATE_DIM;
  out_msg.layout.dim[0].stride = 1;
  out_msg.layout.data_offset = 0;

  out_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
  out_msg.layout.dim[1].label = "variance";
  out_msg.layout.dim[1].size = STATE_DIM;
  out_msg.layout.dim[1].stride = 1;

  for (size_t i = 0; i < STATE_DIM; ++i)
  {
    out_msg.data[i] = state[i].mean;
    out_msg.data[STATE_DIM + i] = state[i].variance;
  }
  publisher_->publish(out_msg);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathFinderNode>());
  rclcpp::shutdown();
  return 0;
}
