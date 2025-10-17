
import cv2
import sys
import os
import time
import csv
import numpy as np
from PIL import Image
from argparse import ArgumentParser
sys.path.append('../..')
from inference.scene_seg_infer import SceneSegNetworkInfer

FPS=5
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


def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True, help="Path to PyTorch checkpoint file")
    parser.add_argument("-i", "--input_folder", required=True, help="Path to folder containing .png images")
    parser.add_argument("-o", "--output_csv", default="inference_times.csv", help="Output CSV filename for inference timings")
    parser.add_argument("-v", "--video_output_path", help="Optional: Path to save the semantic output video (.avi)")
    args = parser.parse_args()

    print(f"[INFO] Loading SceneSeg model from: {args.model_checkpoint_path}")
    model = SceneSegNetworkInfer(checkpoint_path=args.model_checkpoint_path)
    print("[INFO] Model loaded successfully")

    image_files = sorted([f for f in os.listdir(args.input_folder) if f.endswith(".png")])
    if not image_files:
        print("[ERROR] No .png images found in the input folder.")
        return

    print(f"[INFO] Found {len(image_files)} images to process.")
    alpha = 0.5
    timings = []

    # Initialize video writer if needed
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
        start = time.perf_counter()
        prediction = model.inference(image_pil)
        end = time.perf_counter()
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

    # Write results to CSV
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
    
    
    
    
    
    
