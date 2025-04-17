## trace_pytorch_model.py
A script to load a pytorch model from among the AutoSeg Vision Foundation Model networks via a .pth checkpoint file, trace and then export the traced model as a *.pt file.

### Example Usage
```bash
  python3 trace_pytorch_model.py -n SceneSeg -p /path/to/SceneSeg/weights.pth -o /path/to/SceneSeg_Export/traced_model.pt
```

### convert_pytorch_to_onnx.py
A script to load a pytorch model from among the AutoSeg Vision Foundation Model networks and convert and export that model to ONNX format at FP32 precision

### Example Usage
```bash
  python3 convert_pytorch_to_onnx.py -n SceneSeg -p /path/to/SceneSeg/weights.pth -o /path/to/SceneSeg_Export/converted_model.onnx
```

