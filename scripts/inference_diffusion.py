"""Evaluation pipeline for the trained diffusion action head.

Samples random episodes/timesteps from the diffusion dataset, runs reverse
flow matching inference with inpainting, and generates a Markdown report
with per-arm trajectory tables, MAE metrics, and joint angle comparison plots.

Usage:
    python -m scripts.inference_diffusion \
        --dit_checkpoint outputs/diffusion/checkpoint.pt \
        --adapter_dir outputs/experiments/run_*/checkpoints \
        --vlm_cache_dir outputs/vlm_cache

    python -m scripts.inference_diffusion \
        --dit_checkpoint outputs/diffusion/checkpoint.pt \
        --adapter_dir outputs/experiments/run_*/checkpoints \
        --num_samples 20 --seed 123
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.diffusion_model import ActionDiT1D
from model.diffusion_dataset import (
    build_combined_diffusion_dataset,
    DIFFUSION_HORIZON,
    INPAINT_FRAMES,
)
from model.vlm_encoder import FrozenVLMEncoder


def denormalize_actions(
    normalized: np.ndarray,
    joint_mins: np.ndarray,
    joint_maxs: np.ndarray,
) -> np.ndarray:
    """Convert from [-1, 1] back to degrees.

    Args:
        normalized: (..., 6) array of normalized action values in [-1, 1].
        joint_mins: (6,) array of joint minimum values in degrees.
        joint_maxs: (6,) array of joint maximum values in degrees.

    Returns:
        (..., 6) array of joint values in degrees.
    """
    return (normalized + 1.0) / 2.0 * (joint_maxs - joint_mins) + joint_mins


def diffusion_inference(
    model: ActionDiT1D,
    own_vlm_emb: torch.Tensor,
    cross_vlm_emb: torch.Tensor,
    gt_actions_normalized: torch.Tensor,
    inpaint_mask: torch.Tensor,
    num_steps: int = 50,
    device: str = "cuda",
) -> torch.Tensor:
    """Run reverse flow matching inference with inpainting.

    Starts from pure noise at t=1 and integrates backward to t=0 using Euler
    steps. Frames marked by inpaint_mask are clamped to ground truth at every
    step.

    Args:
        model: Trained ActionDiT1D model.
        own_vlm_emb: (1, 2048) VLM embedding for this arm.
        cross_vlm_emb: (1, 2048) VLM embedding for the other arm.
        gt_actions_normalized: (1, 40, 6) normalized GT actions in [-1, 1].
        inpaint_mask: (40,) bool tensor, True for GT/inpaint frames.
        num_steps: Number of Euler integration steps.
        device: Device to run on.

    Returns:
        (1, 40, 6) tensor of predicted actions in [-1, 1].
    """
    dt = 1.0 / num_steps

    # Start from pure noise
    x = torch.randn(1, DIFFUSION_HORIZON, 6, device=device)

    # Set inpainted frames to GT
    x[:, inpaint_mask, :] = gt_actions_normalized[:, inpaint_mask, :]

    for step in range(num_steps):
        t = 1.0 - step * dt  # goes from 1.0 to dt
        t_tensor = torch.tensor([t], device=device)
        v_pred = model(x, t_tensor, own_vlm_emb, cross_vlm_emb)
        x = x - v_pred * dt
        # Re-apply inpainting
        x[:, inpaint_mask, :] = gt_actions_normalized[:, inpaint_mask, :]

    return x


def plot_joint_comparison_diffusion(
    gt: np.ndarray,
    predicted: np.ndarray,
    arm: str,
    save_path: Path,
    inpaint_frames: int = 16,
) -> None:
    """Plot GT vs predicted joint angles, with vertical line at inpaint boundary.

    Args:
        gt: (40, 6) ground truth joint angles in degrees.
        predicted: (40, 6) predicted joint angles in degrees.
        arm: "left" or "right".
        save_path: Path to save the figure.
        inpaint_frames: Number of frames that were inpainted (GT-clamped).
    """
    timesteps = np.arange(gt.shape[0])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"{arm.capitalize()} Arm — Joint Angles: GT vs Diffusion Predicted",
        fontsize=14,
    )

    for j, ax in enumerate(axes.flat):
        ax.plot(
            timesteps, gt[:, j], "o-",
            label="Ground Truth", color="tab:blue", markersize=3,
        )
        ax.plot(
            timesteps, predicted[:, j], "s--",
            label="Predicted", color="tab:orange", markersize=3,
        )
        ax.axvline(
            x=inpaint_frames - 0.5, color="red", linestyle=":",
            alpha=0.7, label="Inpaint boundary",
        )
        ax.axvspan(
            0, inpaint_frames - 0.5, alpha=0.1, color="green",
            label="GT inpaint" if j == 0 else None,
        )
        ax.set_title(f"Joint {j}")
        ax.set_xlabel("Frame (8 Hz)")
        ax.set_ylabel("Angle (deg)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def array_to_md_table(arr: np.ndarray) -> str:
    """Convert an (T, 6) array to a Markdown table with timestep rows and joint columns."""
    header = "| t | j0 | j1 | j2 | j3 | j4 | j5 |"
    sep = "|---|------|------|------|------|------|------|"
    rows = [header, sep]
    for t in range(arr.shape[0]):
        vals = " | ".join(f"{arr[t, j]:.2f}" for j in range(arr.shape[1]))
        rows.append(f"| {t} | {vals} |")
    return "\n".join(rows)


def evaluate_samples(
    model: ActionDiT1D,
    vlm_encoder: FrozenVLMEncoder | None,
    dataset,
    output_dir: Path,
    num_samples: int = 10,
    num_steps: int = 50,
    seed: int = 42,
) -> None:
    """Sample random episodes/timesteps, run diffusion inference, and write report.

    Args:
        model: Trained ActionDiT1D model on CUDA.
        vlm_encoder: FrozenVLMEncoder for live encoding, or None if using cached embeddings.
        dataset: ConcatDataset of BimanualDiffusionDatasets.
        output_dir: Directory to write results and report.
        num_samples: Number of random samples to evaluate.
        num_steps: Number of Euler integration steps.
        seed: Random seed for reproducibility.
    """
    device = next(model.parameters()).device

    # Get action tokenizer for denormalization
    tokenizer = dataset.datasets[0].action_tokenizer
    joint_mins = tokenizer.joint_mins  # (6,)
    joint_maxs = tokenizer.joint_maxs  # (6,)

    # Sample random indices
    rng = np.random.RandomState(seed)
    total_samples = len(dataset)
    sample_indices = rng.choice(
        total_samples, size=min(num_samples, total_samples), replace=False,
    )
    sample_indices.sort()

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sample_num, dataset_idx in enumerate(sample_indices):
        sample = dataset[dataset_idx]
        ep_idx = sample["episode_idx"]
        t_start = sample["t_8hz_start"]

        # Determine which sub-dataset this sample came from
        cumulative = 0
        dataset_name = "unknown"
        for ds in dataset.datasets:
            if dataset_idx < cumulative + len(ds):
                dataset_name = ds.dataset_root.name
                break
            cumulative += len(ds)

        print(
            f"\n--- Sample {sample_num:02d}: episode {ep_idx}, "
            f"t_8hz_start {t_start} ({dataset_name}) ---"
        )

        sample_dir = output_dir / f"sample_{sample_num:02d}"
        sample_dir.mkdir(exist_ok=True)

        # Get VLM embeddings
        if "left_vlm_emb" in sample:
            left_vlm_emb = sample["left_vlm_emb"].unsqueeze(0).to(device).float()
            right_vlm_emb = sample["right_vlm_emb"].unsqueeze(0).to(device).float()
        else:
            left_vlm_emb = vlm_encoder.encode(
                [sample["left_vlm_messages"]]
            ).to(device).float()
            right_vlm_emb = vlm_encoder.encode(
                [sample["right_vlm_messages"]]
            ).to(device).float()

        # Get GT actions (normalized, both arms)
        left_actions_norm = (
            torch.from_numpy(sample["left_actions_8hz"])
            .unsqueeze(0).to(device).float()
        )  # (1, 40, 6)
        right_actions_norm = (
            torch.from_numpy(sample["right_actions_8hz"])
            .unsqueeze(0).to(device).float()
        )  # (1, 40, 6)

        # Inpaint mask
        inpaint_mask = torch.from_numpy(sample["inpaint_mask"]).to(device)  # (40,)

        arm_results = {}
        for arm in ("left", "right"):
            if arm == "left":
                own_vlm_emb = left_vlm_emb
                cross_vlm_emb = right_vlm_emb
                gt_norm = left_actions_norm
            else:
                own_vlm_emb = right_vlm_emb
                cross_vlm_emb = left_vlm_emb
                gt_norm = right_actions_norm

            # Run diffusion inference
            with torch.no_grad():
                predicted_norm = diffusion_inference(
                    model, own_vlm_emb, cross_vlm_emb,
                    gt_norm, inpaint_mask,
                    num_steps=num_steps, device=device,
                )

            # Move to numpy
            predicted_norm_np = predicted_norm.squeeze(0).cpu().numpy()  # (40, 6)
            gt_norm_np = gt_norm.squeeze(0).cpu().numpy()  # (40, 6)

            # Denormalize to degrees
            predicted_deg = denormalize_actions(predicted_norm_np, joint_mins, joint_maxs)
            gt_deg = denormalize_actions(gt_norm_np, joint_mins, joint_maxs)

            # Compute MAE on predicted region only (frames INPAINT_FRAMES to end)
            pred_region_pred = predicted_deg[INPAINT_FRAMES:]
            pred_region_gt = gt_deg[INPAINT_FRAMES:]
            mae = float(np.abs(pred_region_pred - pred_region_gt).mean())

            print(f"  {arm:5s} MAE (frames {INPAINT_FRAMES}-{DIFFUSION_HORIZON - 1}): {mae:.2f} degrees")

            # Save numpy arrays
            np.save(sample_dir / f"{arm}_predicted_deg.npy", predicted_deg)
            np.save(sample_dir / f"{arm}_gt_deg.npy", gt_deg)

            # Save plot
            plot_joint_comparison_diffusion(
                gt_deg, predicted_deg, arm,
                sample_dir / f"{arm}_joints.png",
                inpaint_frames=INPAINT_FRAMES,
            )

            arm_results[arm] = {
                "predicted_deg": predicted_deg,
                "gt_deg": gt_deg,
                "mae": mae,
            }

        results.append({
            "sample_num": sample_num,
            "ep_idx": ep_idx,
            "t_8hz_start": t_start,
            "dataset_name": dataset_name,
            "arms": arm_results,
        })

    _write_report(results, output_dir)
    print(f"\nReport written to {output_dir / 'report.md'}")


def _write_report(results: list[dict], output_dir: Path) -> None:
    """Write the Markdown report summarizing all diffusion inference samples."""
    lines = [
        "# Diffusion Action Head Evaluation Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Samples:** {len(results)}",
        f"**Horizon:** {DIFFUSION_HORIZON} frames at 8 Hz (5 seconds)",
        f"**Inpaint frames:** 0-{INPAINT_FRAMES - 1} ({INPAINT_FRAMES} frames, GT clamped)",
        f"**Predicted frames:** {INPAINT_FRAMES}-{DIFFUSION_HORIZON - 1} ({DIFFUSION_HORIZON - INPAINT_FRAMES} frames)",
        "",
        "---",
        "",
    ]

    for r in results:
        n = r["sample_num"]
        lines.append(
            f"## Sample {n:02d} — Episode {r['ep_idx']}, "
            f"t_8hz_start {r['t_8hz_start']} (dataset: {r['dataset_name']})"
        )
        lines.append("")

        for arm in ("left", "right"):
            arm_data = r["arms"][arm]
            gt_deg = arm_data["gt_deg"]
            predicted_deg = arm_data["predicted_deg"]

            lines.append(f"### {arm.capitalize()} Arm")
            lines.append("")

            lines.append("**Ground Truth Trajectory (40 frames, degrees):**")
            lines.append(array_to_md_table(gt_deg))
            lines.append("")

            lines.append(
                f"**Predicted Trajectory (frames {INPAINT_FRAMES}-{DIFFUSION_HORIZON - 1}, degrees):**"
            )
            lines.append(array_to_md_table(predicted_deg[INPAINT_FRAMES:]))
            lines.append("")

            lines.append(
                f"**MAE (frames {INPAINT_FRAMES}-{DIFFUSION_HORIZON - 1}):** "
                f"{arm_data['mae']:.2f} degrees"
            )
            lines.append("")

            lines.append("**Joint Angle Comparison:**")
            lines.append(f"![{arm} joints](sample_{n:02d}/{arm}_joints.png)")
            lines.append("")

        # Per-sample comparison table
        left_mae = r["arms"]["left"]["mae"]
        right_mae = r["arms"]["right"]["mae"]
        lines.append("### Left vs Right Comparison")
        lines.append("")
        lines.append("| Metric | Left Arm | Right Arm |")
        lines.append("|--------|----------|-----------|")
        lines.append(f"| MAE    | {left_mae:.2f} deg | {right_mae:.2f} deg |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Sample | Dataset | Episode | t_8hz_start | Left MAE | Right MAE |")
    lines.append("|--------|---------|---------|-------------|----------|-----------|")
    for r in results:
        n = r["sample_num"]
        left_mae = r["arms"]["left"]["mae"]
        right_mae = r["arms"]["right"]["mae"]
        lines.append(
            f"| {n:02d} | {r['dataset_name']} | {r['ep_idx']} | {r['t_8hz_start']} "
            f"| {left_mae:.2f} deg | {right_mae:.2f} deg |"
        )

    if results:
        avg_left = np.mean([r["arms"]["left"]["mae"] for r in results])
        avg_right = np.mean([r["arms"]["right"]["mae"] for r in results])
        lines.append(
            f"| **Avg** | | | | **{avg_left:.2f} deg** | **{avg_right:.2f} deg** |"
        )
    lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline for the trained diffusion action head."
    )
    parser.add_argument(
        "--dit_checkpoint", type=str, required=True,
        help="Path to trained DiT checkpoint .pt file",
    )
    parser.add_argument(
        "--adapter_dir", type=str, required=True,
        help="Path to VLM LoRA adapter",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
    )
    parser.add_argument(
        "--vlm_cache_dir", type=str, default=None,
        help="If set, use cached VLM embeddings",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10,
    )
    parser.add_argument(
        "--num_steps", type=int, default=50,
        help="Number of Euler steps",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/diffusion_inference",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load DiT model from checkpoint
    print(f"Loading DiT checkpoint: {args.dit_checkpoint}")
    checkpoint = torch.load(args.dit_checkpoint, map_location="cpu", weights_only=True)
    dit_model = ActionDiT1D()
    dit_model.load_state_dict(checkpoint["model_state_dict"])
    dit_model.to(device)
    dit_model.eval()
    print(f"DiT model loaded ({sum(p.numel() for p in dit_model.parameters()):,} parameters)")

    # Load VLM encoder (only if not using cached embeddings)
    vlm_encoder = None
    if args.vlm_cache_dir is None:
        print("No VLM cache dir specified, loading VLM encoder for live encoding...")
        vlm_encoder = FrozenVLMEncoder(
            adapter_dir=args.adapter_dir,
            data_dir=args.data_dir,
        )
        print("VLM encoder loaded.")
    else:
        print(f"Using cached VLM embeddings from: {args.vlm_cache_dir}")

    # Load dataset
    print(f"\nBuilding combined diffusion dataset from {args.data_dir}...")
    dataset = build_combined_diffusion_dataset(
        args.data_dir,
        vlm_cache_dir=args.vlm_cache_dir,
    )

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Run evaluation
    evaluate_samples(
        model=dit_model,
        vlm_encoder=vlm_encoder,
        dataset=dataset,
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
