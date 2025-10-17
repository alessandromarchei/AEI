#!/usr/bin/env python3
import os
import cv2
import csv
import time
import glob
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

# ========= Utils =========

def dump_engine(engine: trt.ICudaEngine, profile_idx: int = 0):
    print("=== TensorRT Engine Info (v3 tensors API) ===")
    print(f"# optimization profiles: {engine.num_optimization_profiles}")
    print(f"# I/O tensors: {engine.num_io_tensors}")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        loc = engine.get_tensor_location(name)
        shape = engine.get_tensor_shape(name)  # may contain -1 for dynamic
        kind = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
        print(f"\n[{i}] {name}")
        print(f"  kind: {kind}, dtype: {dtype}, location: {loc}")
        print(f"  engine-shape: {tuple(shape)}")
        # If dynamic, show profile ranges (if available)
        try:
            min_s, opt_s, max_s = engine.get_tensor_profile_shape(name, profile_idx)
            if any(d == -1 for d in shape) or (min_s != opt_s or opt_s != max_s):
                print(f"  profile[{profile_idx}] min:{tuple(min_s)} opt:{tuple(opt_s)} max:{tuple(max_s)}")
        except Exception:
            pass
    print("=============================================\n")


def nchw_or_nhwc_from_shape(shape_tuple):
    # heuristics: return ("NCHW", (N,C,H,W)) or ("NHWC", (N,H,W,C))
    # if ambiguous, default to NCHW
    t = tuple(shape_tuple)
    # Ignore -1s while deciding channel position
    try:
        if len(t) == 4:
            if t[1] == 3:
                return "NCHW"
            if t[3] == 3:
                return "NHWC"
    except Exception:
        pass
    return "NCHW"


def preprocess_image_to_input(img_bgr, layout, target_hw, dtype_np):
    # img_bgr: HxWx3 uint8 (OpenCV)
    Ht, Wt = target_hw
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0  # normalize to [0,1]

    if layout == "NCHW":
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # -> NCHW
    else:
        img = np.expand_dims(img, axis=0)   # -> NHWC

    if dtype_np == np.float16:
        img = img.astype(np.float16)
    elif dtype_np == np.float32:
        img = img.astype(np.float32)
    elif dtype_np == np.int8:
        img = (img * 255.0 - 128).astype(np.int8)  # naive example; real INT8 needs calibration/scale
    else:
        img = img.astype(dtype_np)

    return img


def postprocess_first_output_to_mask(output_arr, layout_guess="NCHW"):
    """
    Very generic: if output is [1, C, H, W] -> argmax over C.
    If output is [1, H, W] -> squeeze.
    If [1, H, W, C] -> argmax over C last.
    Otherwise: best-effort squeeze/reshape.
    """
    arr = output_arr
    if arr.ndim == 4:
        # try NCHW vs NHWC
        if layout_guess == "NCHW" and arr.shape[1] > 1:
            mask = np.argmax(arr, axis=1).squeeze()
        elif layout_guess == "NHWC" and arr.shape[-1] > 1:
            mask = np.argmax(arr, axis=-1).squeeze()
        else:
            # single channel; squeeze N and C
            mask = arr.squeeze()
    elif arr.ndim == 3:
        mask = arr.squeeze()
    else:
        mask = arr
    # compress to uint8 for visualization
    mask = np.asarray(mask, dtype=np.uint8)
    return mask


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

# ========= Runner =========

def run_inference_folder(engine_path, input_folder, output_csv, video_output_path=None, profile_idx=0):
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if not hasattr(engine, "num_io_tensors"):
        raise RuntimeError("This script requires the v3 tensors API (TensorRT ≥ 8.5).")

    dump_engine(engine, profile_idx=profile_idx)

    # Collect I/O tensor names
    input_names = [engine.get_tensor_name(i)
                   for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    output_names = [engine.get_tensor_name(i)
                    for i in range(engine.num_io_tensors)
                    if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

    if len(input_names) != 1:
        raise RuntimeError(f"Expected a single image input, found {len(input_names)} inputs: {input_names}")

    input_name = input_names[0]
    input_dtype_trt = engine.get_tensor_dtype(input_name)
    input_dtype_np = trt.nptype(input_dtype_trt)

    # Decide layout and target size from profile's OPT shape (fallback to engine shape)
    try:
        min_s, opt_s, max_s = engine.get_tensor_profile_shape(input_name, profile_idx)
        in_shape = tuple(opt_s)
    except Exception:
        in_shape = tuple(engine.get_tensor_shape(input_name))

    if len(in_shape) != 4:
        raise RuntimeError(f"Unsupported input rank {len(in_shape)} for {input_name}: {in_shape}")

    layout = nchw_or_nhwc_from_shape(in_shape)
    if layout == "NCHW":
        N, C, Ht, Wt = (1, 3,
                        in_shape[2] if in_shape[2] != -1 else 320,
                        in_shape[3] if in_shape[3] != -1 else 640)
    else:
        N, Ht, Wt, C = (1,
                        in_shape[1] if in_shape[1] != -1 else 320,
                        in_shape[2] if in_shape[2] != -1 else 640,
                        3)

    # Sanity checks
    if C not in (1, 3):
        print(f"[WARN] Channel count is {C}; continuing anyway.")

    # Gather images
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.png")))
    if not image_files:
        raise FileNotFoundError(f"No .png images found in {input_folder}")

    # Create context and stream
    context = engine.create_execution_context()
    stream = cuda.Stream()

    # If multiple profiles, pick one
    if engine.num_optimization_profiles > 1:
        # For TRT 8.6+, set_optimization_profile_async exists; fallback to sync if not
        if hasattr(context, "set_optimization_profile_async"):
            context.set_optimization_profile_async(profile_idx, stream.handle)
        else:
            context.set_optimization_profile(profile_idx)

    # Fix the actual input shape for N=1
    if layout == "NCHW":
        context.set_input_shape(input_name, (1, C, Ht, Wt))
    else:
        context.set_input_shape(input_name, (1, Ht, Wt, C))

    # Concrete shapes and buffer allocations
    # Input
    in_concrete_shape = tuple(context.get_tensor_shape(input_name))
    in_size = trt.volume(in_concrete_shape)
    h_in = cuda.pagelocked_empty(in_size, input_dtype_np)
    d_in = cuda.mem_alloc(h_in.nbytes)
    context.set_tensor_address(input_name, int(d_in))

    # Outputs
    outputs = []
    for name in output_names:
        out_shape = tuple(context.get_tensor_shape(name))
        if any(d == -1 for d in out_shape):
            raise RuntimeError(f"Output tensor {name} still has dynamic dims after setting input shape: {out_shape}")
        out_dtype_np = trt.nptype(engine.get_tensor_dtype(name))
        out_size = trt.volume(out_shape)
        h_out = cuda.pagelocked_empty(out_size, out_dtype_np)
        d_out = cuda.mem_alloc(h_out.nbytes)
        context.set_tensor_address(name, int(d_out))
        outputs.append((name, out_shape, out_dtype_np, h_out, d_out))

    # Set up video writer if requested
    writer = None
    if video_output_path:
        # derive size from first original image
        frame0 = cv2.imread(image_files[0], cv2.IMREAD_COLOR)
        if frame0 is None:
            raise RuntimeError(f"Unable to read first image {image_files[0]}")
        H0, W0 = frame0.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(video_output_path, fourcc, 5, (W0, H0))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer at {video_output_path}")
        print(f"[INFO] Semantic video will be saved to: {video_output_path}")

    # Warmup (optional)
    dummy = np.zeros((Ht, Wt, 3), dtype=np.uint8)
    x = preprocess_image_to_input(dummy, layout, (Ht, Wt), input_dtype_np)
    np.copyto(h_in, x.ravel())
    cuda.memcpy_htod_async(d_in, h_in, stream)
    context.execute_async_v3(stream.handle)
    for _, _, _, h_out, d_out in outputs:
        cuda.memcpy_dtoh_async(h_out, d_out, stream)
    stream.synchronize()

    # Process all images
    timings = []
    print(f"[INFO] Found {len(image_files)} images to process.")
    for img_path in image_files:
        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[WARN] Skipping unreadable image: {os.path.basename(img_path)}")
            continue

        # Preprocess
        x = preprocess_image_to_input(frame, layout, (Ht, Wt), input_dtype_np)

        # Copy to input buffer
        np.copyto(h_in, x.ravel())
        cuda.memcpy_htod_async(d_in, h_in, stream)

        # Inference
        t0 = time.perf_counter()
        context.execute_async_v3(stream.handle)

        # Copy outputs back
        for _, _, _, h_out, d_out in outputs:
            cuda.memcpy_dtoh_async(h_out, d_out, stream)
        stream.synchronize()
        t1 = time.perf_counter()

        dt_ms = (t1 - t0) * 1000.0
        timings.append((os.path.basename(img_path), dt_ms))
        print(f"[TIMING] {os.path.basename(img_path)}: {dt_ms:.2f} ms")

        # Optional visualization to video
        if writer:
            # Use first output for a simple mask visualization
            name0, shape0, _, h0, _ = outputs[0]
            out0 = np.array(h0).reshape(shape0)
            mask = postprocess_first_output_to_mask(out0, layout_guess="NCHW")
            vis = make_visualization(mask)
            vis = cv2.resize(vis, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            blended = cv2.addWeighted(vis, 0.5, frame, 0.5, 0.0)
            writer.write(blended)

    if writer:
        writer.release()
        print(f"[INFO] Video saved to {video_output_path}")

    # CSV + average
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "inference_time_ms"])
        for name, t in timings:
            w.writerow([name, f"{t:.2f}"])

    avg_ms = sum(t for _, t in timings) / max(1, len(timings))
    print(f"\n[INFO] Inference times written to: {output_csv}")
    print(f"[AVERAGE] {avg_ms:.2f} ms per image")


# ========= CLI =========

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--engine", required=True, help="Path to TensorRT engine (.engine/.trt)")
    p.add_argument("-i", "--input_folder", required=True, help="Folder containing .png images")
    p.add_argument("-o", "--output_csv", default="inference_times.csv", help="CSV for per-image timings")
    p.add_argument("-v", "--video_output_path", default=None, help="Optional path to save semantic output video (.avi)")
    p.add_argument("--profile", type=int, default=0, help="Optimization profile index (default: 0)")
    args = p.parse_args()

    run_inference_folder(
        engine_path=args.engine,
        input_folder=args.input_folder,
        output_csv=args.output_csv,
        video_output_path=args.video_output_path,
        profile_idx=args.profile,
    )

if __name__ == "__main__":
    main()
