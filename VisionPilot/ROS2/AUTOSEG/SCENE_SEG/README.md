# autoware_pov_scene_seg

This ROS2 package performs real-time semantic segmentation on camera images using optimized inference backends. It supports both ONNX Runtime and TensorRT backends with various precision modes for maximum performance flexibility.

## Features

- **Dual Backend Support**: Choose between ONNX Runtime and TensorRT
- **Multiple Precision Modes**: FP32, FP16, and INT8 quantized models
- **Engine Caching**: TensorRT engines are automatically cached for faster startup
- **Real-time Performance**: Optimized for high-FPS inference with latency monitoring
- **Flexible Input**: Works with live cameras or video files

## Architecture

The package contains two main components:

1. **SceneSegNode**: A C++ composable node with pluggable inference backends
2. **video_publisher.py**: A Python script for testing with video files

### Inference Backends

#### ONNX Runtime Backend
- **Best for**: Quantized (INT8) models, rapid prototyping, cross-platform compatibility
- **Precision options**: 
  - `"cpu"`: CPU inference
  - `"cuda"`: GPU inference with FP32 precision
- **Supports**: Asymmetric quantization from PyTorch QAT

#### TensorRT Backend  
- **Best for**: Maximum performance on NVIDIA GPUs
- **Precision options**:
  - `"fp32"`: Full precision (baseline accuracy)
  - `"fp16"`: Half precision (~2x faster, minimal accuracy loss)
- **Engine Caching**: Automatically creates and reuses optimized engines
- **File naming**: `model.onnx.{precision}.engine`

## Quick Start

### 1. Build the Package

```bash
# Build with ONNX Runtime and TensorRT support
colcon build --packages-select autoware_pov_scene_seg \
  --cmake-args \
  -DONNXRUNTIME_ROOTDIR=/path/to/your/onnxruntime \
  -DCMAKE_BUILD_TYPE=Release
```

### 2. Test with Video File

```bash
# Source your workspace
source install/setup.bash

# Set library paths for both ONNX Runtime and TensorRT
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:/path/to/tensorrt/lib:$LD_LIBRARY_PATH

# Run test pipeline
ros2 launch autoware_pov_scene_seg test_pipeline.launch.py \
  model_path:=data/your_model.onnx \
  video_path:=data/test_video.mp4 \
  backend:=tensorrt \
  precision:=fp16 \
  measure_latency:=true
```

## Usage Examples

### TensorRT with FP16 (Recommended for Performance)

```bash
ros2 launch autoware_pov_scene_seg test_pipeline.launch.py \
  model_path:=data/SceneSeg_FP32.onnx \
  backend:=tensorrt \
  precision:=fp16 \
  measure_latency:=true
```

**What happens:**
1. First run: Builds FP16 engine from FP32 ONNX model (takes 1-2 minutes)
2. Saves engine as `data/SceneSeg_FP32.onnx.fp16.engine`
3. Subsequent runs: Loads pre-built engine instantly
4. Expected performance: ~2x faster than FP32

### TensorRT with FP32 (Maximum Accuracy)

```bash
ros2 launch autoware_pov_scene_seg test_pipeline.launch.py \
  model_path:=data/SceneSeg_FP32.onnx \
  backend:=tensorrt \
  precision:=fp32 \
  measure_latency:=true
```

**Engine file:** `data/SceneSeg_FP32.onnx.fp32.engine`

### ONNX Runtime with Quantized Model (INT8)

```bash
ros2 launch autoware_pov_scene_seg test_pipeline.launch.py \
  model_path:=data/SceneSeg_INT8.onnx \
  backend:=onnxruntime \
  precision:=cuda \
  measure_latency:=true
```

**Best for:** Models quantized with PyTorch QAT (Quantization Aware Training)

### ONNX Runtime CPU Fallback

```bash
ros2 launch autoware_pov_scene_seg test_pipeline.launch.py \
  model_path:=data/SceneSeg_FP32.onnx \
  backend:=onnxruntime \
  precision:=cpu \
  measure_latency:=false
```

## Live Camera Usage

For real-world deployment with a live camera:

```bash
ros2 launch autoware_pov_scene_seg scene_seg.launch.py \
  model_path:=/path/to/your/model.onnx \
  backend:=tensorrt \
  precision:=fp16 \
  input_image_topic:=/sensing/camera/front/image_raw \
  measure_latency:=true
```

## Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | - | Path to ONNX model file |
| `backend` | `"onnxruntime"` | `"onnxruntime"` or `"tensorrt"` |
| `precision` | `"cpu"` | See precision options below |
| `gpu_id` | `0` | GPU device ID for inference |
| `measure_latency` | `false` | Enable latency monitoring (samples every 200 frames) |
| `input_image_topic` | `"/image_raw"` | Input camera topic |

### Precision Options by Backend

**ONNX Runtime:**
- `"cpu"`: CPU inference
- `"cuda"`: GPU inference (FP32)

**TensorRT:**
- `"fp32"`: Full precision GPU inference
- `"fp16"`: Half precision GPU inference

## Performance Tips

### Backend Selection Guide

**Use TensorRT when:**
- You need maximum performance
- Running FP32 or FP16 models
- Deploying on NVIDIA GPUs
- Inference speed is critical

**Use ONNX Runtime when:**
- Using quantized (INT8) models
- Need cross-platform compatibility
- Rapid prototyping and testing
- CPU inference required

### Engine Caching (TensorRT)

TensorRT automatically optimizes models for your specific hardware:

```
# First run with fp16
[tensorrt_backend]: No pre-built fp16 engine found. Building from ONNX model: data/model.onnx
[tensorrt_backend]: Building TensorRT engine with FP16 precision
[tensorrt_backend]: Saving fp16 engine to data/model.onnx.fp16.engine

# Subsequent runs with fp16  
[tensorrt_backend]: Found pre-built fp16 engine at data/model.onnx.fp16.engine
[tensorrt_backend]: Using TensorRT backend
```

Each precision gets its own engine file:
- `model.onnx.fp32.engine`
- `model.onnx.fp16.engine`

### Latency Monitoring

When `measure_latency:=true`, the node samples inference time every 200 frames:

```
Frame 200: Inference Latency: 14.34 ms (69.8 FPS)
Frame 400: Inference Latency: 13.92 ms (71.8 FPS)
```

## Visualization

View the segmentation output:

```bash
# List available topics
ros2 topic list

# View colorized mask
rqt_image_view ~/out/color_mask

# View raw segmentation mask  
rqt_image_view ~/out/mask
```

## Troubleshooting

### Common Issues

**"Failed to load library":**
```bash
# Ensure library paths are set correctly
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:/path/to/tensorrt/lib:$LD_LIBRARY_PATH
```

**"No optimization profile defined":**
- This happens with dynamic ONNX models. The TensorRT backend automatically handles this.

**"Non-zero zero point not supported":**
- Use ONNX Runtime backend for asymmetrically quantized models
- Use TensorRT backend for FP32/FP16 models only

### Performance Benchmarking

Compare different configurations:

```bash
# Benchmark FP32
ros2 launch autoware_pov_scene_seg test_pipeline.launch.py \
  backend:=tensorrt precision:=fp32 measure_latency:=true

# Benchmark FP16  
ros2 launch autoware_pov_scene_seg test_pipeline.launch.py \
  backend:=tensorrt precision:=fp16 measure_latency:=true

# Benchmark INT8
ros2 launch autoware_pov_scene_seg test_pipeline.launch.py \
  model_path:=data/quantized_model.onnx \
  backend:=onnxruntime precision:=cuda measure_latency:=true
```

## Model Requirements

- **Input**: RGB images (any resolution, automatically resized)
- **Output**: Segmentation logits [batch, classes, height, width]
- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Supported formats**: ONNX models exported from PyTorch, TensorFlow, etc.
