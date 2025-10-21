import os
import sys
import cv2
import time
import csv
import numpy as np
from PIL import Image
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ------------------------
# Helper functions
# ------------------------
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

# ------------------------
# TensorRT Inference class
# ------------------------
class TensorRTInfer:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            binding_shape = self.engine.get_binding_shape(binding)
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = int(np.prod(binding_shape)) * np.dtype(binding_dtype).itemsize

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(int(np.prod(binding_shape)), binding_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape})

    def infer(self, image):
        # Copy image to pagelocked host memory
        np.copyto(self.inputs[0]['host'], image.ravel())

        # Transfer to device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # Return reshaped output
        output = self.outputs[0]['host']
        return output.reshape(self.outputs[0]['shape'])

# ------------------------
# Main script logic
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", required=True, help="Path to TensorRT engine (.trt)")
    parser.add_argument("-i", "--input_folder", required=True, help="Path to folder containing .png images")
    parser.add_argument("-o", "--output_csv", default="inference_times.csv", help="Output CSV filename for inference timings")
    parser.add_argument("-v", "--video_output_path", help="Optional: Path to save the semantic output video (.avi)")
    args = parser.parse_args()

    print(f"[INFO] Loading TensorRT engine from: {args.engine}")
    model = TensorRTInfer(args.engine)
    print("[INFO] Engine loaded successfully")

    image_files = sorted([f for f in os.listdir(args.input_folder) if f.endswith(".png")])
    if not image_files:
        print("[ERROR] No .png images found in the input folder.")
        return

    print(f"[INFO] Found {len(image_files)} images to process.")
    alpha = 0.5
    timings = []

    # Prepare video writer
    video_writer = None
    if args.video_output_path:
        first_image_path = os.path.join(args.input_folder, image_files[0])
        first_frame = cv2.imread(first_image_path, cv2.IMREAD_COLOR)
        height, width = first_frame.shape[:2]
        video_writer = cv2.VideoWriter(
            args.video_output_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            5,
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
        image_resized = cv2.resize(image_rgb, (640, 320))  # Adjust to your model input
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_nchw = np.transpose(image_normalized, (2, 0, 1))
        image_nchw = np.expand_dims(image_nchw, axis=0)

        print(f"[INFO] Running inference on: {image_file}")
        start = time.perf_counter()
        prediction = model.infer(image_nchw)
        end = time.perf_counter()
        duration_ms = (end - start) * 1000
        timings.append((image_file, duration_ms))

        print(f"[TIMING] {image_file}: {duration_ms:.2f} ms")

        # Assuming prediction shape [1, H, W], convert to 2D mask
        pred_mask = prediction.squeeze().astype(np.uint8)

        if video_writer:
            vis_obj = make_visualization(pred_mask)
            vis_obj = cv2.resize(vis_obj, (frame.shape[1], frame.shape[0]))
            blended = cv2.addWeighted(vis_obj, alpha, frame, 1 - alpha, 0)
            video_writer.write(blended)

    if video_writer:
        video_writer.release()
        print(f"[INFO] Video saved to {args.video_output_path}")

    # Write results to CSV
    with open(args.output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "inference_time_ms"])
        for name, t in timings:
            writer.writerow([name, f"{t:.2f}"])

    print(f"\n[INFO] Inference times written to: {args.output_csv}")
    avg_time = sum(t for _, t in timings) / len(timings)
    print(f"[AVERAGE] {avg_time:.2f} ms per image")

if __name__ == "__main__":
    main()
