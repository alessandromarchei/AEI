#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import cv2
import sys
import numpy as np
import time
from PIL import Image
from argparse import ArgumentParser
sys.path.append('../..')
from inference.scene_seg_infer import SceneSegNetworkInfer


def make_visualization(prediction):
    shape = prediction.shape
    row, col = shape[0], shape[1]
    vis_predict_object = np.zeros((row, col, 3), dtype="uint8")

    vis_predict_object[:, :, 0] = 255
    vis_predict_object[:, :, 1] = 93
    vis_predict_object[:, :, 2] = 61

    foreground_labels = np.where(prediction == 1)
    vis_predict_object[foreground_labels[0], foreground_labels[1], 0] = 145
    vis_predict_object[foreground_labels[0], foreground_labels[1], 1] = 28
    vis_predict_object[foreground_labels[0], foreground_labels[1], 2] = 255

    return vis_predict_object

def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-i", "--video_filepath", dest="video_filepath", help="path to input video which will be processed by SceneSeg")
    parser.add_argument("-o", "--output_file", dest="output_file", help="path to output video visualization file, must include output file name")
    parser.add_argument('-v', "--vis", action='store_true', help="flag for whether to show frame by frame visualization while processing is occuring")
    args = parser.parse_args()

    print("[INFO] Loading SceneSeg model...")
    model = SceneSegNetworkInfer(checkpoint_path=args.model_checkpoint_path)
    print("[INFO] SceneSeg model loaded successfully")

    cap = cv2.VideoCapture(args.video_filepath)
    output_filepath_obj = args.output_file + '.avi'
    writer_obj = cv2.VideoWriter(output_filepath_obj,
                                 cv2.VideoWriter_fourcc(*"MJPG"),
                                 25, (1280, 720))

    if not cap.isOpened():
        print("[ERROR] Could not open video file.")
        return
    else:
        print("[INFO] Video file opened. Starting frame processing...")

    alpha = 0.5
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] No more frames to read. Exiting.")
            break

        frame_idx += 1
        print(f"[INFO] Processing frame {frame_idx}")

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image).resize((640, 320))

        # Inference with timing
        start_time = time.perf_counter()
        prediction = model.inference(image_pil)
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        print(f"[TIMING] Inference time for frame {frame_idx}: {inference_time_ms:.2f} ms")

        vis_obj = make_visualization(prediction)

        frame = cv2.resize(frame, (1280, 720))
        vis_obj = cv2.resize(vis_obj, (1280, 720))
        image_vis_obj = cv2.addWeighted(vis_obj, alpha, frame, 1 - alpha, 0)

        if args.vis:
            cv2.imshow('Prediction Objects', image_vis_obj)
            cv2.waitKey(10)

        writer_obj.write(image_vis_obj)

    cap.release()
    writer_obj.release()
    cv2.destroyAllWindows()
    print("[INFO] Video processing completed.")

if __name__ == '__main__':
    main()
# %%
