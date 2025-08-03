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

  publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("output_topic", 10);

  subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "input_topic", 10, std::bind(&PathFinderNode::topic_callback, this, std::placeholders::_1));

  timer_ = this->create_wall_timer(std::chrono::milliseconds(1000), std::bind(&PathFinderNode::timer_callback_, this));

  std::string package_share_dir = ament_index_cpp::get_package_share_directory("PATHFINDER");
  std::string yaml_fp = package_share_dir + "/test/000001/1616005402699.yaml";
  std::string homo_fp = package_share_dir + "/test/image_to_world_transform.yaml";
  H = loadHFromYaml(homo_fp);

  auto egoLanesPts = loadLanesFromYaml(yaml_fp, H);
  std::vector<fittedCurve> egoPaths;
  if (egoLanesPts.size() < 2)
    std::cerr << "Not enough lanes to calculate ego path" << std::endl;

  auto egoLanesPtsLR = {egoLanesPts[2], egoLanesPts[1]}; // identify lane pairs

  std::vector<fittedCurve> egoLanes;
  for (auto lanePts : egoLanesPtsLR)
  {
    std::array<double, 3> coeff = fitQuadPoly(lanePts.BevPoints);
    egoLanes.emplace_back(fittedCurve(coeff));
  }

  drivCorr = drivingCorridor(egoLanes[0], egoLanes[1], std::nullopt);

  RCLCPP_INFO(this->get_logger(), "PathFinder Node started");
}

void PathFinderNode::timer_callback_()
{
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

  auto out_msg = std_msgs::msg::Float64MultiArray();
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
