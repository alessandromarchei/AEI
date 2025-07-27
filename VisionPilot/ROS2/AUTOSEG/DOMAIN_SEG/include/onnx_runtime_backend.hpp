#pragma once

#include "inference_backend.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace autoware_pov::AutoSeg::DomainSeg
{

class OnnxRuntimeBackend : public InferenceBackend
{
public:
  OnnxRuntimeBackend(const std::string & model_path, const std::string & precision, int gpu_id);
  ~OnnxRuntimeBackend() = default;

  bool doInference(const cv::Mat & input_image) override;
  void getDomainMask(cv::Mat & domain_mask, const cv::Size & output_size) const override;
  
  int getModelInputHeight() const override { return model_input_height_; }
  int getModelInputWidth() const override { return model_input_width_; }

private:
  void initializeOrtSession(const std::string & model_path, const std::string & precision, int gpu_id);
  void preprocess(const cv::Mat & input_image, std::vector<float> & output_tensor, std::vector<int64_t> & input_dims);
  
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> ort_session_;
  Ort::MemoryInfo memory_info_;
  
  std::vector<char*> input_names_;
  std::vector<char*> output_names_;
  std::vector<Ort::Value> last_output_tensors_;
  
  int model_input_height_;
  int model_input_width_;
};

}  // namespace autoware_pov::AutoSeg::DomainSeg 