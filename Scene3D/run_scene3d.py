import cv2
import sys
import os
import numpy as np
import torch
from PIL import Image
from argparse import ArgumentParser

from network.inference.scene_3d_infer import Scene3DNetworkInfer,Scene3DOnnxInfer, SceneSegTrtInfer
from network.model_components.scene_3d_network import Scene3DNetwork
from network.model_components.scene_seg_network import SceneSegNetwork

from utils.visualize import visualize_scene3d
from utils.preprocessing import load_image


def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True,
                        help="Path to model (.pth, .onnx, or .trt)")
    parser.add_argument("-i", "--input_folder", required=True,
                        help="Path to folder containing .png images")
    parser.add_argument("-r", "--results_dir", default="results/Scene3D",
                        help="Base results directory (default: results/Scene3D)")
    parser.add_argument("-s", "--suffix", default="",
                        help="Optional suffix for output folder name")
    args = parser.parse_args()

    model_path = args.model_checkpoint_path
    ext = os.path.splitext(model_path)[1].lower()

    #used to convert the original pth from autoware to onnx with including the preprocessing nodes inside the model
    # if ext == ".pth":
    #     print(f"[INFO] Loading PyTorch model from: {model_path}")

    #     # Build + load weights exactly like your working FP32 path
    #     sceneSegNetwork = SceneSegNetwork()
    #     model = Scene3DNetwork(sceneSegNetwork)
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model.to(device)

    #     # Load Scene3D weights (these should include the backbone weights too)
    #     state = torch.load(model_path, map_location=device)
    #     if isinstance(state, dict) and "state_dict" in state:
    #         state = state["state_dict"]
    #     missing, unexpected = model.load_state_dict(state, strict=False)
    #     if missing:   print("[WARN] missing:", missing)
    #     if unexpected:print("[WARN] unexpected:", unexpected)

    #     model.eval()  # IMPORTANT

    #     # Dummy input in [0,1] because your forward expects [0,1] then normalizes
    #     dummy_input = torch.rand(1, 3, 320, 640, dtype=torch.float32, device=device)

    #     onnx_out = os.path.splitext(model_path)[0] + ".onnx"
    #     torch.onnx.export(
    #         model,
    #         dummy_input,
    #         onnx_out,
    #         input_names=["input"],
    #         output_names=["depth"],
    #         opset_version=17,
    #         do_constant_folding=True,
    #         dynamic_axes={"input": {0: "batch"}, "depth": {0: "batch"}}
    #     )

    #     print(f"[INFO] ONNX export completed: {onnx_out}")
    #     sys.exit(0)

    if ext == ".pth":
        print(f"[INFO] Loading PyTorch model from: {model_path}")
        model = Scene3DNetworkInfer(checkpoint_path=model_path)
        infer_fn = model.inference
    elif ext == ".onnx":
        print(f"[INFO] Loading ONNX model from: {model_path}")
        model = Scene3DOnnxInfer(model_path)
        infer_fn = model.inference
    elif ext == ".trt":
        print(f"[INFO] Loading TensorRT engine from: {model_path}")
        model = SceneSegTrtInfer(model_path)
        infer_fn = model.inference
    else:
        print("[ERROR] Unsupported model format. Use .pth, .onnx or .trt")
        return

    print("[INFO] Model loaded successfully")

    image_files = sorted([f for f in os.listdir(args.input_folder) if f.endswith(".png")])
    if not image_files:
        print("[ERROR] No .png images found in the input folder.")
        return

    # Get dataset + scene names
    scene_name = os.path.basename(os.path.normpath(args.input_folder))
    dataset_name = os.path.basename(os.path.dirname(os.path.normpath(args.input_folder)))

    out_folder = os.path.join(
        args.results_dir,
        dataset_name,
        scene_name + (f"_{args.suffix}" if args.suffix else "")
    )
    os.makedirs(out_folder, exist_ok=True)

    print(f"[INFO] Found {len(image_files)} images to process.")
    print(f"[INFO] Results will be saved in: {out_folder}")

    alpha = 0.97  # Blending factor

    for idx, image_file in enumerate(image_files):
        image_pil, frame = load_image(args.input_folder, image_file)

        print(f"[INFO] Running inference on: {image_file}")
        prediction = infer_fn(image_pil)

        # Create mask and overlay on original frame
        masked_output = visualize_scene3d(frame, prediction, alpha)

        # Save PNG
        save_path = os.path.join(out_folder, os.path.splitext(image_file)[0] + ".png")
        cv2.imwrite(save_path, masked_output)

    print(f"\n[INFO] All blended visualizations saved in: {out_folder}")


if __name__ == '__main__':
    main()
