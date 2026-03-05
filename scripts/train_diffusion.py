"""Train a flow-matching diffusion policy for bimanual action prediction.

Two-phase usage:
    # Phase 1: Pre-compute VLM embeddings
    python -m scripts.train_diffusion --precompute_vlm --adapter_dir ... --data_dir data

    # Phase 2: Train diffusion model
    python -m scripts.train_diffusion --adapter_dir ... --vlm_cache_dir outputs/vlm_cache --data_dir data
"""

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.diffusion_model import ActionDiT1D
from model.diffusion_dataset import build_combined_diffusion_dataset
from model.vlm_encoder import FrozenVLMEncoder


def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float) -> float:
    """Linear warmup followed by cosine annealing learning rate schedule.

    Args:
        step: Current training step.
        warmup_steps: Number of warmup steps with linear ramp.
        max_steps: Total number of training steps.
        base_lr: Peak learning rate after warmup.

    Returns:
        Learning rate for the given step.
    """
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def diffusion_collate_fn(batch):
    """Collate cached-embedding diffusion samples into tensors."""
    return {
        "left_actions_8hz": torch.stack([torch.as_tensor(s["left_actions_8hz"]) for s in batch]),
        "right_actions_8hz": torch.stack([torch.as_tensor(s["right_actions_8hz"]) for s in batch]),
        "left_vlm_emb": torch.stack([s["left_vlm_emb"] for s in batch]),
        "right_vlm_emb": torch.stack([s["right_vlm_emb"] for s in batch]),
        "inpaint_mask": torch.stack([torch.as_tensor(s["inpaint_mask"]) for s in batch]),
    }


def precompute_vlm_embeddings(args):
    """Phase 1: Pre-compute and cache VLM embeddings for all dataset samples.

    Loads the frozen VLM encoder and iterates through each sub-dataset in the
    combined diffusion dataset, encoding VLM messages and saving embeddings
    to disk for use during training.

    Args:
        args: Parsed command-line arguments with adapter_dir, data_dir,
              and vlm_cache_dir fields.
    """
    print("=" * 60)
    print("Phase 1: Pre-computing VLM embeddings")
    print("=" * 60)

    # Load VLM encoder
    print(f"\nLoading VLM encoder from adapter: {args.adapter_dir}")
    encoder = FrozenVLMEncoder(
        adapter_dir=args.adapter_dir,
        data_dir=args.data_dir,
    )

    # Build dataset without cache (returns VLM messages)
    print(f"\nBuilding dataset from: {args.data_dir}")
    combined_dataset = build_combined_diffusion_dataset(
        data_dir=args.data_dir,
        vlm_cache_dir=None,
    )

    # Process each sub-dataset
    cache_dir = Path(args.vlm_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCache directory: {cache_dir}")

    for i, sub_dataset in enumerate(combined_dataset.datasets):
        print(f"\nProcessing sub-dataset {i + 1}/{len(combined_dataset.datasets)}: "
              f"{sub_dataset.dataset_root.name}")
        encoder.precompute_and_cache(sub_dataset, cache_dir)

    print("\nVLM embedding precomputation complete.")


def train(args):
    """Phase 2: Train the flow-matching diffusion model.

    Loads cached VLM embeddings and trains an ActionDiT1D model using
    flow matching with inpainting on the ground-truth action prefix.

    Args:
        args: Parsed command-line arguments with all training hyperparameters.
    """
    print("=" * 60)
    print("Phase 2: Training diffusion model")
    print("=" * 60)

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / "diffusion_runs" / f"run_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build dataset with cached embeddings
    print(f"\nBuilding dataset from: {args.data_dir}")
    print(f"VLM cache directory: {args.vlm_cache_dir}")
    dataset = build_combined_diffusion_dataset(
        data_dir=args.data_dir,
        vlm_cache_dir=args.vlm_cache_dir,
    )
    print(f"Total training samples: {len(dataset)}")

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=diffusion_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Batch size: {args.batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Model
    model = ActionDiT1D()
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]
        print(f"Resumed at step {start_step}")

    # Training loop
    print(f"\nTraining for {args.max_steps} steps")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Gradient clipping: {args.grad_clip}")
    print(f"Save every: {args.save_every} steps")
    print(f"Log every: {args.log_every} steps")
    print()

    model.train()
    running_loss = 0.0
    running_count = 0
    step = start_step
    start_time = time.time()
    data_iter = iter(dataloader)

    while step < args.max_steps:
        # Get next batch, cycling through epochs
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to device
        left_actions = batch["left_actions_8hz"].to(device, dtype=torch.float32)     # (B, 40, 6)
        right_actions = batch["right_actions_8hz"].to(device, dtype=torch.float32)   # (B, 40, 6)
        left_vlm_emb = batch["left_vlm_emb"].to(device, dtype=torch.float32)        # (B, 2048)
        right_vlm_emb = batch["right_vlm_emb"].to(device, dtype=torch.float32)      # (B, 2048)
        inpaint_mask = batch["inpaint_mask"].to(device)                               # (B, 40)

        B = left_actions.shape[0]

        # Stack both arms into 2B batch
        # For left arm: own=left_vlm, cross=right_vlm
        # For right arm: own=right_vlm, cross=left_vlm
        x_0 = torch.cat([left_actions, right_actions], dim=0)        # (2B, 40, 6)
        own_vlm = torch.cat([left_vlm_emb, right_vlm_emb], dim=0)   # (2B, 2048)
        cross_vlm = torch.cat([right_vlm_emb, left_vlm_emb], dim=0) # (2B, 2048)
        inpaint = torch.cat([inpaint_mask, inpaint_mask], dim=0)     # (2B, 40)

        # Sample timestep t ~ U(0, 1)
        t = torch.rand(2 * B, 1, 1, device=device)  # (2B, 1, 1)

        # Sample noise
        epsilon = torch.randn_like(x_0)  # (2B, 40, 6)

        # Interpolate: x_t = (1 - t) * x_0 + t * epsilon
        x_t = (1.0 - t) * x_0 + t * epsilon

        # Inpainting: replace x_t where mask is True with clean x_0
        inpaint_expanded = inpaint.unsqueeze(-1)  # (2B, 40, 1)
        x_t = torch.where(inpaint_expanded, x_0, x_t)

        # Predict velocity
        v_pred = model(x_t, t.squeeze(-1).squeeze(-1), own_vlm, cross_vlm)  # (2B, 40, 6)

        # Target velocity: v_target = epsilon - x_0
        v_target = epsilon - x_0

        # Loss: MSE on predicted frames only (where inpaint mask is False)
        pred_mask = ~inpaint_expanded  # (2B, 40, 1)
        loss = ((v_pred - v_target) ** 2 * pred_mask).sum() / pred_mask.sum()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Update learning rate
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Optimizer step
        optimizer.step()

        # Update running stats
        loss_val = loss.item()
        running_loss += loss_val
        running_count += 1
        step += 1

        # Logging
        if step % args.log_every == 0:
            avg_loss = running_loss / running_count
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0.0
            print(
                f"Step {step:>7d}/{args.max_steps} | "
                f"loss: {loss_val:.6f} | "
                f"avg_loss: {avg_loss:.6f} | "
                f"lr: {lr:.2e} | "
                f"grad_norm: {grad_norm:.4f} | "
                f"steps/s: {steps_per_sec:.2f}"
            )

        # Save checkpoint
        if step % args.save_every == 0:
            ckpt_path = checkpoint_dir / f"step_{step:06d}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final checkpoint
    final_ckpt_path = checkpoint_dir / f"step_{step:06d}.pt"
    if not final_ckpt_path.exists():
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
            },
            final_ckpt_path,
        )
        print(f"Saved final checkpoint: {final_ckpt_path}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete. Total time: {elapsed / 3600:.2f} hours")
    print(f"Final loss: {running_loss / running_count:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute_vlm", action="store_true")
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--vlm_cache_dir", type=str, default="outputs/vlm_cache")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.precompute_vlm:
        precompute_vlm_embeddings(args)
    else:
        train(args)
