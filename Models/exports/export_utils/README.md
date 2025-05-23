# ONNX Model Quantization and Benchmarking Tools

This folder adds two new utilities for ONNX model optimization and performance analysis:

1. `quantize_onnx_model.py`: Converts FP32 ONNX models to INT8 quantized versions
2. `benchmark_onnx_models.py`: Benchmarks and compares FP32 and INT8 model performance

## Features

### Quantization Tool
- Converts FP32 models to INT8 using post-training static quantization
- Uses QDQ (QuantizeLinear/DeQuantizeLinear) format for optimal compatibility
- Supports per-channel quantization for weights
- Includes comprehensive calibration data handling
- Maintains float32 precision during preprocessing

### Benchmarking Tool
- Measures inference speed, model size, and memory usage
- Compares outputs between FP32 and INT8 models
- Supports visualization of model outputs and differences
- Uses real calibration data for accurate benchmarking
- Provides detailed performance metrics and statistics

## FP32 Model Pre-processing

Before quantization, it's recommended to pre-process your FP32 model to ensure optimal results. This involves several steps that improve quantization effectiveness:

1. **Symbolic Shape Inference**: Particularly helpful for transformer models
2. **ONNX Runtime Model Optimization**: Combines operations for better performance
3. **ONNX Shape Inference**: Ensures all tensor shapes are available for quantization

Pre-processing can be performed using ONNX Runtime's built-in tools:

```bash
python -m onnxruntime.quantization.preprocess \
    --input /path/to/model.onnx \
    --output /path/to/model-infer.onnx
```

For scene segmentation models, this pre-processing helps by:
- Merging operations like Convolution+BatchNormalization 
- Adding shape information required for accurate quantization
- Optimizing model structure for better inference performance

After pre-processing, the resulting model (with suffix "-infer") can be used as input for the quantization tool.

For more details on quantization techniques, see the [ONNX Runtime Quantization Examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu).

## Calibration Dataset Structure
The calibration tools for scene segmentation models expect the dataset to be organized in the following structure:

```
SceneSeg/
├── ACDC/
│   └── ACDC/
│       └── images/
│           ├── image1.jpg
│           └── ...
├── BDD100K/
│   └── BDD100K/
│       └── images/
│           ├── image1.jpg
│           └── ...
├── comma10k/
│   └── comma10k/
│       └── images/
│           ├── image1.jpg
│           └── ...
├── IDDAW/
│   └── IDDAW/
│       └── images/
│           ├── image1.jpg
│           └── ...
├── Mapillary_Vistas/
│   └── Mapillary_Vistas/
│       └── images/
│           ├── image1.jpg
│           └── ...
└── MUSES/
    └── MUSES/
        └── images/
            ├── image1.jpg
            └── ...
```

Each dataset should contain RGB images in JPG or PNG format that are representative of real driving scenes. The tools will automatically sample from these directories for calibration and benchmarking.

## Usage

### 1. Model Quantization

Convert your FP32 model to INT8:

```bash
python quantize_onnx_model.py \
    --input_model /path/to/fp32_model.onnx \
    --output_model /path/to/int8_model.onnx \
    --calibration_data_dir /path/to/datasets/
```

### 2. Model Benchmarking

Compare FP32 and INT8 model performance:

```bash
python benchmark_onnx_models.py \
    --fp32_model /path/to/fp32_model.onnx \
    --int8_model /path/to/int8_model.onnx \
    --dataset /path/to/datasets/ \
    --save_outputs
```

## Notes


- **Reduce model size**: INT8 quantization typically reduces model size by 75%
- **Improve inference speed**: Up to 4x speedup on compatible hardware
- **Maintain accuracy**: Minimal impact on model output quality (verified with output comparison)
- **Standardize workflow**: Consistent process for quantizing and benchmarking scene segmentation models

The implementation provides:
- Post-training static quantization with representative dataset calibration
- QDQ format for optimal compatibility across hardware
- Comprehensive benchmarking with performance metrics
- Output comparison for quality assessment
- Configurable parameters with sensible defaults

These tools have been thoroughly tested with scene segmentation models across multiple hardware configurations and dataset combinations.

## Dependencies
- onnx
- onnxruntime
- numpy
- opencv-python
- psutil
- matplotlib

## Testing
The tools have been tested with:
- Various ONNX models
- Different input sizes
- Multiple calibration datasets
- Different hardware configurations

## Notes
- Quantization may affect model accuracy
- Performance improvements vary by hardware
- Calibration data should be representative of real-world usage

## trace_pytorch_model.py
A script to load a pytorch model from among the AutoSeg Vision Foundation Model networks via a .pth checkpoint file, trace and then export the traced model as a *.pt file.

### Example Usage
```bash
  python3 trace_pytorch_model.py -n SceneSeg -p /path/to/SceneSeg/weights.pth -o /path/to/SceneSeg_Export/traced_model.pt
```

### convert_pytorch_to_onnx.py
A script to load a pytorch model from among the AutoSeg Vision Foundation Model networks and convert and export that model to ONNX format at FP32 precision as a *.onnx file.

### Example Usage
```bash
  python3 convert_pytorch_to_onnx.py -n SceneSeg -p /path/to/SceneSeg/weights.pth -o /path/to/SceneSeg_Export/converted_model.onnx
```

