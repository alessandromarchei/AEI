#include "visualize_masks_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <rclcpp_components/register_node_macro.hpp>

namespace autoware_pov::visualization
{

VisualizeMasksNode::VisualizeMasksNode(const rclcpp::NodeOptions & options)
: Node("visualize_masks_node", options)
{
  // Parameters
  const std::string image_topic = this->declare_parameter<std::string>("image_topic");
  const std::string mask_topic = this->declare_parameter<std::string>("mask_topic");
  viz_type_ = this->declare_parameter<std::string>("viz_type", "scene");
  const std::string output_topic = this->declare_parameter<std::string>("output_topic", "~/out/image");
  measure_latency_ = this->declare_parameter<bool>("measure_latency", true);

  createColorMap(viz_type_);

  // Publisher
  pub_ = image_transport::create_publisher(this, output_topic);

  // Subscribers
  sub_image_.subscribe(this, image_topic, rmw_qos_profile_sensor_data);
  sub_mask_.subscribe(this, mask_topic, rmw_qos_profile_sensor_data);

  // Synchronizer
  sync_ = std::make_shared<Synchronizer>(SyncPolicy(10), sub_image_, sub_mask_);
  sync_->registerCallback(std::bind(&VisualizeMasksNode::onData, this, std::placeholders::_1, std::placeholders::_2));
}

void VisualizeMasksNode::createColorMap(const std::string& viz_type)
{
  if (viz_type == "scene") {
    // Match scene_seg.cpp exactly: cv::Vec3b(0,0,0), cv::Vec3b(0,0,255), cv::Vec3b(0,0,0)
    // Class 0: Black background, Class 1: Red foreground, Class 2: Black
    color_map_ = {{0, 0, 0}, {0, 0, 255}, {0, 0, 0}};  // BGR format
  } else if (viz_type == "domain") {
    // Match domain_seg.cpp: RGB(255,93,61) and RGB(28,148,255) - keep as RGB since that's what original used  
    color_map_ = {{255, 93, 61}, {28, 148, 255}};  // RGB format as in original domain_seg.cpp
  } else {
    // Default
    color_map_ = {{0, 0, 255}};
  }
}

void VisualizeMasksNode::onData(const Image::ConstSharedPtr& image_msg, const Image::ConstSharedPtr& mask_msg)
{
  // --- Latency Watcher Start ---
  if (measure_latency_ && (++frame_count_ % LATENCY_SAMPLE_INTERVAL == 0)) {
    viz_start_time_ = std::chrono::steady_clock::now();
  }
  // -----------------------------
  
  cv_bridge::CvImagePtr image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
  cv_bridge::CvImagePtr mask_ptr = cv_bridge::toCvCopy(mask_msg, sensor_msgs::image_encodings::MONO8);

  // Create color mask using the same approach as original scene_seg.cpp and domain_seg.cpp
  cv::Mat color_mask = cv::Mat::zeros(mask_ptr->image.size(), CV_8UC3);
  
  if (viz_type_ == "scene") {
    // Scene segmentation: class IDs (0, 1, 2) 
    for (size_t class_id = 0; class_id < color_map_.size(); ++class_id) {
      cv::Mat class_mask = (mask_ptr->image == class_id);
      color_mask.setTo(color_map_[class_id], class_mask);
    }
  } else if (viz_type_ == "domain") {
    // Domain segmentation: binary (0, 255)
    cv::Mat bg_mask = (mask_ptr->image == 0);
    cv::Mat fg_mask = (mask_ptr->image == 255);
    color_mask.setTo(color_map_[0], bg_mask);  // Background
    color_mask.setTo(color_map_[1], fg_mask);  // Foreground
  }

  // Resize color mask to match input image if needed
  cv::Mat resized_color_mask;
  if (color_mask.size() != image_ptr->image.size()) {
    cv::resize(color_mask, resized_color_mask, image_ptr->image.size());
  } else {
    resized_color_mask = color_mask;
  }

  // Blend exactly like the original code: cv::addWeighted(color_mask, 0.5, original_image, 0.5, 0.0, blended_image)
  cv::Mat blended_image;
  cv::addWeighted(resized_color_mask, 0.5, image_ptr->image, 0.5, 0.0, blended_image);

  sensor_msgs::msg::Image::SharedPtr out_msg =
    cv_bridge::CvImage(image_msg->header, sensor_msgs::image_encodings::BGR8, blended_image).toImageMsg();
  pub_.publish(out_msg);
  
  // --- Latency Watcher End & Report ---
  if (measure_latency_ && (frame_count_ % LATENCY_SAMPLE_INTERVAL == 0)) {
    auto viz_end_time = std::chrono::steady_clock::now();
    auto latency_ms =
      std::chrono::duration<double, std::milli>(viz_end_time - viz_start_time_)
        .count();
    RCLCPP_INFO(
      this->get_logger(), "Frame %zu: Visualization Latency: %.2f ms (%.1f FPS)", frame_count_,
      latency_ms, 1000.0 / latency_ms);
  }
  // ------------------------------------
}

}  // namespace autoware_pov::visualization

RCLCPP_COMPONENTS_REGISTER_NODE(autoware_pov::visualization::VisualizeMasksNode)