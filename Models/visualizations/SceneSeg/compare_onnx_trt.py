#!/usr/bin/env python3
"""
Compare SceneSeg PyTorch (.pth via SceneSegNetworkInfer) vs TensorRT engine.

Usage:
  python compare_sceneseg_pth_trt.py \
    --pth /path/to/checkpoint.pth \
    --engine /path/to/model.engine \
    --input-folder /path/to/images \
    --num-samples 20 \
    --profile 0 \
    --tol-abs 1e-3 --tol-rel 1e-3 \
    --recursive \
    [--repo-root /path/to/repo/root]

Notes:
- We import SceneSegNetworkInfer from inference.scene_seg_infer (like your script).
- Inputs are resized to the TRT engine’s concrete input HxW; we feed the same size to PyTorch.
- We compare the final label masks (pixel accuracy) and also numeric diffs if outputs are logits.
"""

import os
import glob
import argparse
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

import sys

# ---------- Paths & images ----------

def expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

def find_images(root: str, recursive: bool = False) -> List[str]:
    root = expand(root)
    pats = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    imgs = []
    if recursive:
        for p in pats:
            imgs.extend(glob.glob(os.path.join(root, "**", p), recursive=True))
    else:
        for p in pats:
            imgs.extend(glob.glob(os.path.join(root, p)))
    return sorted(imgs)

# ---------- TRT helpers (v3 API) ----------

def dump_trt_io(engine):
    ins, outs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        (ins if mode == trt.TensorIOMode.INPUT else outs).append(name)
    return ins, outs

def guess_layout(shape: Tuple[int, ...]) -> str:
    if len(shape) == 4:
        if shape[1] == 3: return "NCHW"
        if shape[3] == 3: return "NHWC"
    return "NCHW"

def preprocess_for_trt(img_bgr: np.ndarray, layout: str, hw: Tuple[int, int], dtype) -> np.ndarray:
    Ht, Wt = hw
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    if layout == "NCHW":
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, 0)        # NCHW
    else:
        img = np.expand_dims(img, 0)        # NHWC
    return img.astype(dtype, copy=False)

def trt_output_to_mask(arr: np.ndarray) -> np.ndarray:
    """Accepts one TRT output array; returns HxW uint8 mask."""
    a = arr
    if a.ndim == 4:
        # [N,C,H,W] or [N,H,W,C]
        if a.shape[1] > 1:         # NCHW logits
            mask = np.argmax(a, axis=1)[0].astype(np.uint8)
        elif a.shape[-1] > 1:      # NHWC logits
            mask = np.argmax(a, axis=-1)[0].astype(np.uint8)
        else:
            # single-channel: threshold at 0.5
            m = a.reshape(a.shape[0], -1)
            mask = (a[0, 0] > 0.5).astype(np.uint8)
    elif a.ndim == 3:
        # [N,H,W] -> binary
        mask = (a[0] > 0.5).astype(np.uint8)
    elif a.ndim == 2:
        mask = (a > 0.5).astype(np.uint8)
    else:
        # fallback
        mask = (a.ravel() > 0.5).astype(np.uint8)
    return mask

def compare_arrays(a: np.ndarray, b: np.ndarray, eps: float = 1e-12):
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    diff = a - b
    abs_max = float(np.max(np.abs(diff)))
    abs_mean = float(np.mean(np.abs(diff)))
    denom = np.maximum(np.abs(b), eps)
    rel = np.abs(diff) / denom
    rel_max = float(np.max(rel))
    rel_mean = float(np.mean(rel))
    return abs_max, abs_mean, rel_max, rel_mean

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="Path to SceneSeg PyTorch checkpoint (.pth)")
    ap.add_argument("--engine", required=True, help="TensorRT engine (.engine/.trt)")
    ap.add_argument("--input_folder", required=True, help="Folder with images")
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--tol-abs", type=float, default=1e-3)
    ap.add_argument("--tol-rel", type=float, default=1e-3)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--repo-root", type=str, default=None,
                    help="Repo root so we can import inference.scene_seg_infer. "
                         "Default: script_dir/../..")
    args = ap.parse_args()

    # --- Make sure we can import SceneSegNetworkInfer like in your script ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = expand(args.repo_root) if args.repo_root else expand(os.path.join(script_dir, "..", ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)

    try:
        from inference.scene_seg_infer import SceneSegNetworkInfer
    except Exception as e:
        raise RuntimeError(f"Failed to import SceneSegNetworkInfer from '{repo_root}'. "
                           f"Pass --repo-root if needed. Error: {e}")

    # --- Load PyTorch SceneSeg model (weights only, same as your code) ---
    print("[INFO] Loading SceneSeg model...")
    model = SceneSegNetworkInfer(checkpoint_path=expand(args.pth))
    print("[INFO] SceneSeg model loaded.")

    # --- Load TRT engine (v3 tensors API) ---
    logger = trt.Logger(trt.Logger.ERROR)
    with open(expand(args.engine), "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if not hasattr(engine, "num_io_tensors"):
        raise RuntimeError("TensorRT v3 tensors API required (TRT ≥ 8.5)")

    trt_inputs, trt_outputs = dump_trt_io(engine)
    print(f"[TRT] inputs:  {trt_inputs}")
    print(f"[TRT] outputs: {trt_outputs}")
    if len(trt_inputs) != 1:
        raise RuntimeError(f"Expected exactly 1 TRT input, found {len(trt_inputs)}")

    inp_name = trt_inputs[0]
    inp_dtype_trt = engine.get_tensor_dtype(inp_name)
    np_dtype_trt = trt.nptype(inp_dtype_trt)

    # Pick concrete input shape (prefer profile opt)
    try:
        min_s, opt_s, max_s = engine.get_tensor_profile_shape(inp_name, args.profile)
        in_shape = tuple(opt_s)
    except Exception:
        in_shape = tuple(engine.get_tensor_shape(inp_name))

    layout = guess_layout(in_shape)
    if layout == "NCHW":
        N, C, Ht, Wt = 1, (in_shape[1] if in_shape[1] != -1 else 3), (in_shape[2] if in_shape[2] != -1 else 320), (in_shape[3] if in_shape[3] != -1 else 640)
        trt_shape = (N, C, Ht, Wt)
    else:
        N, Ht, Wt, C = 1, (in_shape[1] if in_shape[1] != -1 else 320), (in_shape[2] if in_shape[2] != -1 else 640), (in_shape[3] if in_shape[3] != -1 else 3)
        trt_shape = (N, Ht, Wt, C)

    # --- Setup TRT context/buffers ---
    ctx = engine.create_execution_context()
    stream = cuda.Stream()

    if engine.num_optimization_profiles > 1 and hasattr(ctx, "set_optimization_profile_async"):
        ctx.set_optimization_profile_async(args.profile, stream.handle)
    elif engine.num_optimization_profiles > 1:
        ctx.set_optimization_profile(args.profile)

    ctx.set_input_shape(inp_name, trt_shape)
    in_concrete = tuple(ctx.get_tensor_shape(inp_name))
    in_size = int(trt.volume(in_concrete))
    h_in = cuda.pagelocked_empty(in_size, np_dtype_trt)
    d_in = cuda.mem_alloc(h_in.nbytes)
    ctx.set_tensor_address(inp_name, int(d_in))

    trt_out_bufs = []
    for name in trt_outputs:
        shp = tuple(ctx.get_tensor_shape(name))
        if any(x == -1 for x in shp):
            raise RuntimeError(f"TRT output {name} still dynamic after setting shape: {shp}")
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = int(trt.volume(shp))
        h_out = cuda.pagelocked_empty(size, dtype)
        d_out = cuda.mem_alloc(h_out.nbytes)
        ctx.set_tensor_address(name, int(d_out))
        trt_out_bufs.append((name, shp, dtype, h_out, d_out))

    # --- Images ---
    args.input_folder = expand(args.input_folder)
    print(f"[DEBUG] Looking for images under: {args.input_folder}")
    images = find_images(args.input_folder, recursive=args.recursive)
    print(f"[DEBUG] Found {len(images)} images")
    if not images:
        print("[DEBUG] Directory exists?", os.path.isdir(args.input_folder))
        print("[DEBUG] Absolute path:", os.path.abspath(args.input_folder))
        print("[DEBUG] Sample ls:", os.listdir(args.input_folder)[:10] if os.path.isdir(args.input_folder) else "N/A")
        raise FileNotFoundError("No images found in input folder")
    images = images[:args.num_samples]

    # --- Compare loop ---
    all_ok = True
    tot_pix = 0
    tot_correct = 0

    for p in images:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] skip unreadable {os.path.basename(p)}")
            continue

        # ---------- PyTorch SceneSeg ----------
        # Use the *same size* as TRT input for fairness
        img_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((Wt, Ht))
        pred_pt = model.inference(img_pil)  # expected HxW (labels) or logits
        if isinstance(pred_pt, np.ndarray):
            pass
        else:
            # if tensor or list, convert to numpy
            try:
                pred_pt = np.array(pred_pt)
            except Exception:
                pred_pt = np.asarray(pred_pt)

        # Convert PyTorch output to label mask
        if pred_pt.ndim == 2:            # HxW labels
            mask_pt = pred_pt.astype(np.uint8)
        elif pred_pt.ndim == 3:          # CxHxW logits
            if pred_pt.shape[0] > 1:
                mask_pt = np.argmax(pred_pt, axis=0).astype(np.uint8)
            else:
                mask_pt = (pred_pt[0] > 0.5).astype(np.uint8)
        elif pred_pt.ndim == 4:          # NCHW logits or NHWC
            if pred_pt.shape[1] > 1:
                mask_pt = np.argmax(pred_pt, axis=1)[0].astype(np.uint8)
            else:
                mask_pt = (pred_pt[0,0] > 0.5).astype(np.uint8)
        else:
            mask_pt = np.array(pred_pt).astype(np.uint8)
        if mask_pt.shape != (Ht, Wt):
            mask_pt = cv2.resize(mask_pt, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

        # ---------- TensorRT ----------
        x_trt = preprocess_for_trt(bgr, layout, (Ht, Wt), np_dtype_trt).ravel()
        np.copyto(h_in, x_trt)
        cuda.memcpy_htod_async(d_in, h_in, stream)
        ctx.execute_async_v3(stream_handle=stream.handle)
        for _, _, _, h_out, d_out in trt_out_bufs:
            cuda.memcpy_dtoh_async(h_out, d_out, stream)
        stream.synchronize()

        # first output only for mask
        out0 = np.array(trt_out_bufs[0][3], copy=True).reshape(trt_out_bufs[0][1])
        mask_trt = trt_output_to_mask(out0)
        if mask_trt.shape != (Ht, Wt):
            mask_trt = cv2.resize(mask_trt, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

        # ---------- Compare ----------
        # Pixel accuracy on masks
        common = (mask_pt.shape[0] == mask_trt.shape[0]) and (mask_pt.shape[1] == mask_trt.shape[1])
        if not common:
            print(f"[BAD] {os.path.basename(p)} shape mismatch PT {mask_pt.shape} vs TRT {mask_trt.shape}")
            all_ok = False
            continue

        correct = int((mask_pt == mask_trt).sum())
        total = int(mask_pt.size)
        acc = correct / max(1, total)

        # Optional numeric diff if logits-like shapes are compatible
        abs_max = abs_mean = rel_max = rel_mean = float("nan")
        if pred_pt.ndim in (3,4):
            # try to align shapes to out0 for numeric diff
            pt_logits = pred_pt
            trt_logits = out0
            # Basic alignment attempts
            if pt_logits.shape != trt_logits.shape:
                # common case: NCHW vs NHWC
                if pt_logits.ndim == 4 and trt_logits.ndim == 4:
                    if pt_logits.shape == (trt_logits.shape[0], trt_logits.shape[3], trt_logits.shape[1], trt_logits.shape[2]):
                        trt_logits = np.transpose(trt_logits, (0,3,1,2))
            if pt_logits.shape == trt_logits.shape:
                abs_max, abs_mean, rel_max, rel_mean = compare_arrays(pt_logits.ravel(), trt_logits.ravel())

        ok = (acc == 1.0) or (abs_max <= args.tol_abs) or (rel_max <= args.tol_rel)
        status = "OK " if ok else "BAD"
        if not ok: all_ok = False

        print(f"[{status}] {os.path.basename(p)} | acc={acc*100:.2f}% "
              f"| abs_max={abs_max:.3e} abs_mean={abs_mean:.3e} "
              f"| rel_max={rel_max:.3e} rel_mean={rel_mean:.3e}")

        tot_pix += total
        tot_correct += correct

    # --- Summary ---
    overall_acc = (tot_correct / max(1, tot_pix)) if tot_pix else 0.0
    print("\n[SUMMARY] overall pixel accuracy: {:.2f}%".format(overall_acc * 100.0))
    if all_ok:
        print("[SUMMARY] All samples within tolerance / or exact-match on masks.")
    else:
        print("[SUMMARY] Some samples exceeded tolerances or mismatched masks.")

if __name__ == "__main__":
    main()
