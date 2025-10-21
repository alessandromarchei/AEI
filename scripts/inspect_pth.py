#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

def get_param_summary(state_dict):
    return {k: v.numel() for k, v in state_dict.items()}

def plot_param_pie(param_summary, out_path, top_n=8):
    # Sort by param count
    sorted_params = sorted(param_summary.items(), key=lambda x: x[1], reverse=True)

    # Top N + others
    top_layers = sorted_params[:top_n]
    other_layers = sorted_params[top_n:]
    others_sum = sum(v for _, v in other_layers)
    total_sum = sum(param_summary.values())

    top_params = list(top_layers)
    if others_sum > 0:
        top_params.append(("Others", others_sum))

    labels = [k for k, _ in top_params]
    values = [v for _, v in top_params]
    percent_labels = [f"{k}\n{v / total_sum * 100:.1f}%\n{v:,}" for k, v in top_params]

    colors = cm.get_cmap('viridis')(np.linspace(0.2, 0.9, len(top_params)))

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, _ = ax.pie(values, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.4, edgecolor='white'))

    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(angle)) * 1.3
        y = np.sin(np.deg2rad(angle)) * 1.3
        ax.text(x, y, percent_labels[i], ha='center', va='center', fontsize=9)

    plt.title("Top Layers by Parameter Count", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"📄 Saved visualization to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pth checkpoint")
    parser.add_argument("--top-n", type=int, default=8, help="Number of top layers to display")
    parser.add_argument("--viz", action="store_true", help="Save pie chart as PDF (no interactive plot)")
    args = parser.parse_args()

    # Load checkpoint (to CPU for safety)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # If it's a dict with 'state_dict' key, use that
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Print summary of layers
    print("🔎 Layer summary:")
    for name, tensor in state_dict.items():
        print(f"  {name:50s} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} params={tensor.numel()}")

    # If requested, make PDF
    if args.viz:
        param_summary = get_param_summary(state_dict)
        out_path = Path(args.checkpoint).with_suffix(".params_pie.pdf")
        plot_param_pie(param_summary, out_path, top_n=args.top_n)

if __name__ == "__main__":
    main()
