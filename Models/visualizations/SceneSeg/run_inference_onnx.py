import cv2
import sys
import os
import time
import csv
import numpy as np
from PIL import Image
from argparse import ArgumentParser

# Enable verbose logs to see TensorRT execution
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "0"  # 0=verbose, 1=info, 2=warning, 3=error

import onnxruntime as ort

# Try GPU memory profiling
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    def print_gpu_memory(msg=""):
        free, total = cuda.mem_get_info()
        used = total - free
        print(f"[GPU MEMORY] {msg} Used: {used / 1024**2:.2f} MB / Total: {total / 1024**2:.2f} MB")
except ImportError:
    def print_gpu_memory(msg=""):
        print(f"[GPU MEMORY] {msg} PyCUDA not installed.")

FPS = 5

def make_visualization(prediction):
    shape = prediction.shape
    vis_predict_object = np.zeros((shape[0], shape[1], 3), dtype="uint8")

    vis_predict_object[:, :, 0] = 255
    vis_predict_object[:, :, 1] = 93
    vis_predict_object[:, :, 2] = 61

    fg = np.where(prediction == 1)
    vis_predict_object[fg[0], fg[1], 0] = 145
    vis_predict_object[fg[0], fg[1], 1] = 28
    vis_predict_object[fg[0], fg[1], 2] = 255

    return vis_predict_object

class SceneSegONNXInfer:
    def __init__(self, onnx_path, bit_width=16):

        if bit_width not in [8, 16]:
            raise ValueError("Bit width must be either 8 or 16.")
        
        if bit_width == 8:
            print("[INFO] Using INT8 quantized ONNX model.")
            trt_int8_enable = True
            trt_fp16_enable = False
        else:
            print("[INFO] Using FP16 quantized ONNX model.")
            trt_int8_enable = False
            trt_fp16_enable = True

        max_mem = 500000000

        providers = [
            ('TensorrtExecutionProvider', {
                'trt_engine_cache_enable': True,
                'trt_int8_enable': trt_int8_enable,
                'trt_fp16_enable': trt_fp16_enable,
                'trt_dla_enable': False,
                'trt_engine_cache_enable' : True,
            })
        ]
        print("[INFO] Initializing ONNX Runtime session with TensorRT...")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print("[INFO] ONNX Runtime session initialized successfully.")

    def inference(self, pil_img):
        img_array = np.asarray(pil_img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
        outputs = self.session.run(None, {self.input_name: img_array})
        prediction = outputs[0][0]  # Remove batch dimension
        prediction = np.argmax(prediction, axis=0)  # Class per pixel
        return prediction

def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_onnx_path", required=True, help="Path to the ONNX model file (INT8 quantized)")
    parser.add_argument("-i", "--input_folder", required=True, help="Path to folder containing .png images")
    parser.add_argument("-o", "--output_csv", default="inference_times.csv", help="Output CSV filename for inference timings")
    parser.add_argument("-v", "--video_output_path", help="Optional: Path to save the semantic output video (.avi)")
    parser.add_argument("--bit_width", type=int, default=16, help="Bit width for quantization (default: 16)")
    args = parser.parse_args()


    print(f"[INFO] Loading ONNX model from: {args.model_onnx_path}")
    model = SceneSegONNXInfer(onnx_path=args.model_onnx_path)

    image_files = sorted([f for f in os.listdir(args.input_folder) if f.endswith(".png")])
    if not image_files:
        print("[ERROR] No .png images found in the input folder.")
        return

    print(f"[INFO] Found {len(image_files)} images to process.")
    alpha = 0.5
    timings = []

    video_writer = None
    if args.video_output_path:
        first_image_path = os.path.join(args.input_folder, image_files[0])
        first_frame = cv2.imread(first_image_path, cv2.IMREAD_COLOR)
        height, width = first_frame.shape[:2]
        video_writer = cv2.VideoWriter(
            args.video_output_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            FPS,
            (width, height)
        )
        print(f"[INFO] Semantic video will be saved to: {args.video_output_path}")

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(args.input_folder, image_file)
        frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[WARNING] Skipping unreadable image: {image_file}")
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb).resize((640, 320))

        print(f"[INFO] Running inference on: {image_file}")
        print_gpu_memory("Before inference")
        start = time.perf_counter()
        prediction = model.inference(image_pil)
        end = time.perf_counter()
        print_gpu_memory("After inference")

        duration_ms = (end - start) * 1000
        timings.append((image_file, duration_ms))

        print(f"[TIMING] {image_file}: {duration_ms:.2f} ms")

        if video_writer:
            vis_obj = make_visualization(prediction)
            vis_obj = cv2.resize(vis_obj, (frame.shape[1], frame.shape[0]))
            blended = cv2.addWeighted(vis_obj, alpha, frame, 1 - alpha, 0)
            video_writer.write(blended)

    if video_writer:
        video_writer.release()
        print(f"[INFO] Video saved to {args.video_output_path}")

    with open(args.output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "inference_time_ms"])
        for name, t in timings:
            writer.writerow([name, f"{t:.2f}"])

    print(f"\n[INFO] Inference times written to: {args.output_csv}")
    avg_time = sum(t for _, t in timings) / len(timings)
    print(f"[AVERAGE] {avg_time:.2f} ms per image")

if __name__ == '__main__':
    main()
