#!/usr/bin/env python3
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import cv2
import glob
import argparse

# -------------------------------
# Utility: collect images
# -------------------------------
def collect_calibration_images(data_dirs):
    all_images = []
    for root in [os.path.expanduser(d) for d in data_dirs]:
        # Caso 1: la cartella contiene già immagini
        imgs = glob.glob(os.path.join(root, "*.png")) + glob.glob(os.path.join(root, "*.jpg"))
        if imgs:
            all_images.extend(imgs)
            continue

        # Caso 2: cerca ricorsivamente cartelle che iniziano con "test" o "test_ref"
        for dirpath, dirnames, filenames in os.walk(root):
            base = os.path.basename(dirpath)
            if base.startswith("test"):
                imgs = glob.glob(os.path.join(dirpath, "**", "*.png"), recursive=True)
                imgs += glob.glob(os.path.join(dirpath, "**", "*.jpg"), recursive=True)
                all_images.extend(imgs)

    return all_images


# -------------------------------
# Custom INT8 Calibrator
# -------------------------------
class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_dirs, cache_file, input_shape):
        super(ImageCalibrator, self).__init__()
        self.cache_file = cache_file
        self.input_shape = input_shape  # (C, H, W)
        self.batch_size = 1
        self.use_cache_only = (data_dirs is None or len(data_dirs) == 0)

        if not self.use_cache_only:
            self.image_files = collect_calibration_images(data_dirs)
            if len(self.image_files) == 0:
                raise RuntimeError(f"No images found in calibration folders (searched {data_dirs})")
            print(f"[Calibrator] Found {len(self.image_files)} images from {len(data_dirs)} root(s).")
        else:
            self.image_files = []
            print("[Calibrator] No folders provided, will rely only on existing cache file.")

        self.current_index = 0
        self.device_input = cuda.mem_alloc(
            trt.volume((self.batch_size,) + self.input_shape) * np.float32().nbytes
        )

    def preprocess_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Could not read image {img_path}")
        img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        return img

    def get_batch(self, names):
        if self.use_cache_only:
            return None
        if self.current_index + self.batch_size > len(self.image_files):
            return None
        batch = []
        for i in range(self.batch_size):
            img = self.preprocess_image(self.image_files[self.current_index + i])
            batch.append(img)
        batch = np.ascontiguousarray(batch)
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[Calibrator] Using existing calibration cache: {self.cache_file}")
            return open(self.cache_file, "rb").read()
        return None

    def write_calibration_cache(self, cache):
        if not self.use_cache_only:
            print(f"[Calibrator] Writing calibration cache: {self.cache_file}")
            with open(self.cache_file, "wb") as f:
                f.write(cache)


# -------------------------------
# Build Engine with INT8
# -------------------------------
def build_int8_engine(onnx_path, engine_path, calib_dirs, cache_file, input_shape):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[Parser] Loading ONNX file from {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 * (1 << 20))

    # Optimization profile for dynamic input
    profile = builder.create_optimization_profile()
    profile.set_shape("input",
                      min=(1, 3, 320, 640),
                      opt=(1, 3, 320, 640),
                      max=(1, 3, 320, 640))
    config.add_optimization_profile(profile)

    # Force INT8 mode
    config.set_flag(trt.BuilderFlag.INT8)
    calibrator = ImageCalibrator(calib_dirs, cache_file, input_shape)
    config.int8_calibrator = calibrator

    print("[Builder] Building INT8 engine... this may take a while.")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build INT8 engine")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"[Builder] INT8 engine saved to {engine_path}")


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INT8 Calibration for TensorRT")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--engine", required=True, help="Output path for TensorRT engine")
    parser.add_argument("--calib-cache", default="calib.cache", help="Path for calibration cache file")
    parser.add_argument("--folders", nargs="*", help="List of folders with calibration images (optional)")
    args = parser.parse_args()

    # Fixed input shape from your ONNX model: (3,320,640)
    input_shape = (3, 320, 640)

    build_int8_engine(args.onnx, args.engine, args.folders, args.calib_cache, input_shape)
