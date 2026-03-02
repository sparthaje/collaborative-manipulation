# IMPLEMENTATION.md — Siamese VLA Fine-Tuning Implementation Notes

## Files Created

| File | Task | Purpose |
|------|------|---------|
| `scripts/__init__.py` | — | Makes `scripts` an importable package |
| `scripts/tokenize_actions.py` | Task 1 | Action tokenization (encode/decode continuous ↔ discrete) |
| `scripts/augment_episodes.py` | Task 2 | Generate per-episode prompt templates and 4 FPS indices |
| `scripts/dataloader.py` | Task 3 | `SiameseVLADataset` — PyTorch Dataset yielding chat messages |
| `scripts/train.py` | Task 4 | QLoRA SFT training with TRL |
| `scripts/inference.py` | Task 5 | Load adapter + predict action chunks from images |

## Running the Pipeline

```bash
# 1. Generate augmentation metadata (one-time)
uv run python -m scripts.augment_episodes

# 2. Validate tokenization
uv run python -m scripts.tokenize_actions

# 3. Validate dataloader
uv run python -m scripts.dataloader

# 4. Train (adjust --max_steps for real training)
uv run python -m scripts.train --max_steps 100 --output_dir outputs/siamese_vla

# 5. Inference
uv run python -m scripts.inference --adapter_dir outputs/siamese_vla --episode 0 --frame 50 --arm left
```

## Key Implementation Decisions

### Action Tokenization (Task 1)
- **Shared bin ranges:** Joint i on left and right arms use the same bin boundaries, computed as `min(q01_left[i], q01_right[i])` / `max(q99_left[i], q99_right[i])`.
- **5% margin** added on each side of q01/q99 to handle near-boundary values.
- **Outlier handling:** ~19.4% of dataset actions fall outside the q01/q99+margin range. These get clamped to the nearest bin. This is by design — using min/max would waste resolution on outliers.
- **Token naming:** `<action_j{joint}_b{bin}>` where joint ∈ [0,5] and bin ∈ [0,255]. Total: 1536 tokens.
- **Bin widths:** Range from 0.11° (gripper) to 0.89° (wrist_flex). Max reconstruction error for in-range values: 0.45° (< 1 bin width).

### Episode Augmentation (Task 2)
- **4 FPS resampling:** Uses `round(i * 30 / 4)` for each 4fps index, producing frame indices like [0, 8, 15, 22, 30, ...].
- **Task parsing:** `grabber;picker;black_marker` → "Hand off the black marker (grabber to picker)".
- **Chain-of-thought:** V1 uses empty `<think>\n</think>` block. Can be enhanced later.
- **Output:** 24 JSON files in `data/.../augmentation/`.

### Dataloader (Task 3)
- **Dataset size:** 6502 samples = 2 × 3251 valid frames (left + right per frame).
- **Valid frames:** `num_frames_4fps - 8` per episode (need room for 8-step chunk).
- **Action history clamping:** At episode start, history indices are clamped to 0 (repeats first action).
- **Video loading:** Uses `lerobot.datasets.video_utils.decode_video_frames()` with timestamp-based access. Timestamps are shifted by episode `from_timestamp` since multiple episodes share MP4 files.
- **Image format:** CHW float32 in [0,1], resolution 1080×1920 (native).

### Training (Task 4)
- **Collate function:** Converts torch tensors → PIL Images for the Qwen3-VL processor, then applies `apply_chat_template()` for proper tokenization.
- **`max_length=None`** is critical to avoid truncating image tokens.
- **`skip_prepare_dataset=True`** since we provide a custom collate function.
- **`remove_unused_columns=False`** since our dataset returns message lists, not dicts.
- **Smoke test results:** 2 steps completed in 3s, loss 19.68→19.09, peak GPU memory 6.9 GB on RTX 4090.

### Inference (Task 5)
- **Action token parsing:** Regex extraction of `<action_j\d+_b\d+>` from generated text.
- **Graceful degradation:** If fewer than 48 tokens generated, warns and zero-pads.
- **Two modes:** Dataset sample (for validation) and raw image (for deployment).
- **Vision token budget:** Reduced to max 1024 tokens at inference for speed.

## Validation Summary

| Criterion | Status | Details |
|-----------|--------|---------|
| Data loading end-to-end | ✅ | 6502 samples, correct tensor shapes |
| Siamese pairing doubles data | ✅ | `len(dataset) == 2 * total_valid_frames` |
| Augmentation metadata | ✅ | 24 JSON files, schema validated |
| Action discretization invertible | ✅ | Max error 0.45° < bin width 0.89° (in-range) |
| SFT training runs | ✅ | Loss decreases over 2 steps, no OOM (6.9 GB peak) |
| Inference pipeline works | ✅ | Loads adapter, generates output, parses tokens |

## Known Limitations
- The training collate function processes images one by one through PIL conversion. For large-scale training, consider pre-tokenizing the dataset.
- Empty CoT (`<think></think>`) — enhance with actual reasoning when available.
- After only 2 training steps, the model outputs natural language instead of action tokens (expected). Full training (100+ steps) needed for the model to learn the action token format.
