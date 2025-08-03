#pragma once

#include "path_finder.hpp"
#include "estimator.hpp"

#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

class PathFinderNode : public rclcpp::Node
{
public:
    PathFinderNode();

private:
    void topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void timer_callback_();

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
    Estimator bayesFilter;
    cv::Mat H;
    drivingCorridor drivCorr;
    const double proc_SD = 0.5;
    const double meas_SD = 0.5;
    const double epsilon = 0.05;
};
