#pragma once

#include "path_finder.hpp"
#include "estimator.hpp"

#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

class PathFinderNode : public rclcpp::Node
{
public:
    PathFinderNode();

private:
    void topic_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);//to remove
    void callbackLaneL(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    void callbackLaneR(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    void callbackPath(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr publisher_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_laneL_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_laneR_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_path_;
    std::vector<std::array<float, 2>> reshapeTo2D(const std_msgs::msg::Float32MultiArray::SharedPtr &msg);
    Estimator bayesFilter;
    drivingCorridor drivCorr;
    const double proc_SD = 0.5;
    const double meas_SD = 0.5;
    const double epsilon = 0.05;
};
