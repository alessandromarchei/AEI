# SceneSeg Deployment on Jetson Orin with TensorRT

This repository provides a reproducible workflow for deploying SceneSeg (Autoware) on NVIDIA Jetson Orin using TensorRT. The verified workflow:

- Export PyTorch model ➜ ONNX (FP32, fixed batch size = 1)  
- Convert ONNX ➜ TensorRT Engine (FP32 / FP16 / INT8 with calibration)  
- Run inference / benchmark with trtexec

---

## 1. Requirements

- NVIDIA Jetson Orin with JetPack 6.x (includes CUDA ≥ 12.2, cuDNN, drivers)  
- TensorRT 10.4 (tarball for L4T / aarch64 + CUDA 12.x)  
- CMake ≥ 3.28 (optional tools; trtexec included in TensorRT)  
- Python + PyTorch only for ONNX export

Compatibility and downloads:
- TensorRT Support Matrix: https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html  
- TensorRT Downloads (Jetson): https://developer.nvidia.com/tensorrt  
- trtexec docs: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-tool  
- INT8 sample/calibrator guide: https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html

---

## 2. Install TensorRT 10.4 (tarball)

1. Download the Jetson tarball from NVIDIA:  
    https://developer.nvidia.com/tensorrt

2. Extract to your home directory:
    ```bash
    tar -xvzf TensorRT-10.4.0.xx.l4t.aarch64-gnu.cuda-12.x.tar.gz -C $HOME
    ```

3. Add TensorRT libraries to your environment (add to ~/.bashrc):
    ```bash
    export LD_LIBRARY_PATH="$HOME/TensorRT-10.4.0.xx/lib:$LD_LIBRARY_PATH"
    ```

4. (Optional) Install CUDA compatibility package if required by your JetPack image:
    ```bash
    sudo apt-get update && sudo apt-get install -y cuda-compat-12-6
    ```

trtexec is located in:
```
$HOME/TensorRT-10.4.0.xx/bin/trtexec
```

---

## 3. Export PyTorch ➜ ONNX (FP32, batch=1)

The model must be exported with a fixed batch size = 1. Dynamic axes will cause errors when building TensorRT engines.

Example CLI (script `convert_pytorch_to_onnx.py`):
```bash
python convert_pytorch_to_onnx.py \
  --name SceneSeg \
  -p /path/to/SceneSeg/sceneseg.pth \
  -o onnx/SceneSeg_FP32.onnx
```

Inside the export script, ensure ONNX export uses a fixed-size input and opset 14:
```python
torch.onnx.export(
     model,
     input_data,                   # e.g. tensor shape [1,3,320,640]
     "onnx/SceneSeg_FP32.onnx",
     export_params=True,
     opset_version=14,
     do_constant_folding=True,
     input_names=['input'],
     output_names=['output']
     # Do not use dynamic_axes or dynamic batch size
)
```

Notes:
- Use RGB images for calibration (INT8). If images are grayscale, expand to 3 channels in preprocessing.
- ONNX opset 14 is a safe choice for TensorRT 10.4 compatibility.

---

## 4. Build TensorRT Engine with trtexec

FP32 engine:
```bash
$HOME/TensorRT-10.4.0.xx/bin/trtexec \
  --onnx=onnx/SceneSeg_FP32.onnx \
  --saveEngine=trt/sceneseg_fp32.trt
```

FP16 engine:
```bash
$HOME/TensorRT-10.4.0.xx/bin/trtexec \
  --onnx=onnx/SceneSeg_FP32.onnx \
  --fp16 \
  --saveEngine=trt/sceneseg_fp16.trt
```

INT8 engine (requires calibration):
- Generate a calibration cache using TensorRT sampleINT8 or your own calibrator and dataset.
- Run trtexec with --int8 and --calib:
```bash
$HOME/TensorRT-10.4.0.xx/bin/trtexec \
  --onnx=onnx/SceneSeg_FP32.onnx \
  --int8 \
  --calib=calibration/calib.cache \
  --saveEngine=trt/sceneseg_int8.trt \
  --verbose
```

References:
- INT8 Calibration Guide and TensorRT SampleINT8 (see NVIDIA docs linked above)

---

## 5. Run Inference / Benchmark

Run inference for 10 seconds:
```bash
$HOME/TensorRT-10.4.0.xx/bin/trtexec \
  --loadEngine=trt/sceneseg_fp32.trt \
  --duration=10
```
Repeat for sceneseg_fp16.trt and sceneseg_int8.trt to compare performance.

---

## 6. Performance (input size 3×320×640)

- FP32 → ~8 FPS  
- FP16 → ~21 FPS  
- INT8 → ~36 FPS

---

## 7. Tips & Troubleshooting

- Ensure LD_LIBRARY_PATH points to the TensorRT lib directory used to build/run trtexec.  
- Fixed batch size is mandatory for engine building; avoid dynamic axes in export.  
- For INT8, verify calibration images reflect real deployment input distribution.  
- Use --verbose with trtexec to get detailed builder/runtime logs.

---

Place this README.md content at /home/alessandro/work/AEI/README.md and commit. Adjust paths (TensorRT version, model path, calibration cache) to your environment as needed.