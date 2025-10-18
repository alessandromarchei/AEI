import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically manage CUDA context
import numpy as np
import cv2
import os
import argparse
import sys
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def preprocess_image(img_path, input_shape):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to load image {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = input_shape[-2], input_shape[-1]
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))    # HWC -> CHW
    img = np.expand_dims(img, axis=0)     # Add batch dim
    return img

def run_inference(engine_path, images_folder):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    stream = cuda.Stream()

    inputs, outputs = [], []

    # Allocate buffers for each tensor
    for name in engine:
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = trt.volume(shape)

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append({"name": name, "host": host_mem, "device": device_mem, "shape": shape})
            context.set_tensor_address(name, device_mem)
        else:
            outputs.append({"name": name, "host": host_mem, "device": device_mem, "shape": shape})
            context.set_tensor_address(name, device_mem)

    # Get image paths
    img_paths = sorted([os.path.join(images_folder, f)
                        for f in os.listdir(images_folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not img_paths:
        print(f"[FATAL] No images found in {images_folder}")
        sys.exit(1)

    input_shape = inputs[0]["shape"]

    times = []

    for img_path in img_paths:
        # Preprocess
        input_img = preprocess_image(img_path, input_shape)
        np.copyto(inputs[0]["host"], input_img.ravel())

        # H2D
        cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)

        # Measure inference only
        t0 = time.perf_counter()
        if not context.execute_async_v3(stream.handle):
            raise RuntimeError("Inference failed")
        stream.synchronize()
        t1 = time.perf_counter()
        inf_time = (t1 - t0) * 1000.0  # ms

        # D2H
        for out in outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
        stream.synchronize()

        times.append(inf_time)
        print(f"✅ {os.path.basename(img_path)} | Output shape: {outputs[0]['shape']} | Inference: {inf_time:.3f} ms")

    if times:
        avg = sum(times) / len(times)
        print(f"\n[STATS] {len(times)} images processed")
        print(f"[STATS] Average inference time: {avg:.3f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT Inference Runner")
    parser.add_argument("-e", "--engine", required=True, help="Path to TensorRT engine (.trt/.engine)")
    parser.add_argument("-i", "--input", required=True, help="Path to input folder with images")
    args = parser.parse_args()

    if not os.path.isfile(args.engine):
        print(f"[FATAL] Engine file not found: {args.engine}")
        sys.exit(1)
    if not os.path.isdir(args.input):
        print(f"[FATAL] Input folder not found: {args.input}")
        sys.exit(1)

    run_inference(args.engine, args.input)
