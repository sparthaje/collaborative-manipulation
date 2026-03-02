# DESIGN.md — Unified VLA Fine-Tuning for Bimanual Object Handoff

## Overview

Fine-tune **Cosmos-Reason2-2B** (Qwen3-VL architecture) with QLoRA so that a **single unified policy** controls both arms of a bimanual SO-101 robot during object handoff. The model receives all three camera views (left wrist, right wrist, overhead) plus recent joint history for both arms, and outputs a 2-second action chunk of 12-DoF joint commands (96 tokens = 8 timesteps x 12 joints).

### Change from Original Plan

The original plan used a **Siamese architecture** — two independent copies of the same policy, one per arm, each seeing only its own wrist camera. We replaced this with a unified policy because:

1. **Coordination is the core challenge.** Two independent policies have no mechanism to react to each other. If one arm drifts, the other can't adapt.
2. **2-second open-loop chunks are too long for uncoordinated arms.** A unified policy doesn't need inter-arm communication because it controls both arms simultaneously.
3. **Simpler implementation.** One forward pass, one dataloader, no arm-specific prompting, no Siamese index table.
4. **All 3 cameras give full observability.** The overhead camera provides a global workspace view the original plan discarded.

The tradeoff: we lose the 2x data multiplier from Siamese pairing, and the model must now process 3 images per sample instead of 1 (higher memory/compute per step).

---

## Dataset

**Path:** `data/grabber_picker_black_marker_20260226_211245`

> **Ambiguity resolved:** plan.md referenced a dataset dated `20260228_150311` (24 episodes, 25,724 frames). The actual dataset on disk is dated `20260226_211245` (11 episodes, 11,182 frames at 30 FPS). This design targets the dataset that exists. All scripts accept dataset root as an argument, so switching datasets is trivial.

| Property | Value |
|----------|-------|
| Episodes | 11 |
| Total frames | 11,182 at 30 FPS |
| Action shape | `[12]` — `[left(6), right(6)]` |
| Joint names (per arm) | `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper` |
| Cameras | `left.wrist_left` (1080x1920), `right.wrist_right` (1080x1920), `left.top` (1080x1920) |
| Video codec | AV1, 30 FPS |
| Task | `grabber;picker;black_marker` (single task) |

---

## Architecture

### Input (user message)
```
[system prompt]
[left wrist image] [right wrist image] [overhead image]
"Task: Hand off the black marker (grabber to picker).
Recent joint states (both arms, 12 joints):
t-4: <action_j0_b42> ... <action_j5_b90> | <action_j0_b15> ... <action_j5_b200>
t-3: ...
t-2: ...
t-1: ...
Output the next 8 timesteps of joint commands for both arms as 96 tokens."
```

### Output (assistant message / training target)
```
<think></think>
<action_j0_b42> <action_j1_b130> <action_j2_b64> <action_j3_b200> <action_j4_b15> <action_j5_b90>
<action_j0_b15> <action_j1_b88>  <action_j2_b32> <action_j3_b150> <action_j4_b200> <action_j5_b42>
... (8 timesteps x 12 joints = 96 tokens total)
```

Each timestep outputs 12 tokens: 6 for the left arm joints then 6 for the right arm joints. The same 6 token types are reused for both arms (both are SO-101s with identical joints), and position in the sequence distinguishes left from right.

### Inference-time contract
- **Input:** 3 RGB images (left wrist, right wrist, overhead) + last 4 action vectors (12 floats each) + task prompt
- **Output:** `[8, 12]` float array — 2 seconds of joint commands at 4 FPS for both arms
- **Latency budget:** < 2 seconds (the chunk duration)

---

## Action Tokenization

> **Ambiguity resolved:** With a unified policy we output 12 joints per timestep, but we still only define **6 token types** (one per joint index). Left `shoulder_pan` and right `shoulder_pan` use the same `<action_j0_b*>` tokens because they're physically identical joints with the same range. The output sequence is always ordered `[left_j0..left_j5, right_j0..right_j5]`, so position disambiguates which arm a token belongs to. This keeps vocabulary size at 1,536 tokens (6 joints x 256 bins) instead of 3,072.

### Bin ranges

Per-joint ranges from `meta/stats.json`, using component-wise `min(q01_left, q01_right)` and `max(q99_left, q99_right)` with 5% margin. From the actual dataset stats:

| Joint | q01 range (left, right) | q99 range (left, right) | Shared range (with margin) | Bin width |
|-------|------------------------|------------------------|---------------------------|-----------|
| 0: shoulder_pan | (-17.8, -68.9) | (65.6, 14.7) | ~(-72.2, 68.9) | ~0.55° |
| 1: shoulder_lift | (-103.1, -111.5) | (-6.6, 31.4) | ~(-118.7, 38.6) | ~0.62° |
| 2: elbow_flex | (-34.0, -46.8) | (97.2, 97.3) | ~(-53.9, 104.4) | ~0.62° |
| 3: wrist_flex | (-99.3, -99.1) | (95.6, 101.3) | ~(-109.3, 111.3) | ~0.86° |
| 4: wrist_roll | (-161.4, -169.9) | (-68.7, -86.7) | ~(-174.5, -64.0) | ~0.43° |
| 5: gripper | (1.4, 1.3) | (24.8, 36.7) | ~(-0.4, 38.4) | ~0.15° |

Approximate bin widths range from ~0.15° (gripper) to ~0.86° (wrist_flex). This is sufficient for SO-101 servo resolution.

### Token vocabulary

1,536 special tokens added to the Qwen3-VL tokenizer:
```
<action_j0_b0> ... <action_j0_b255>
<action_j1_b0> ... <action_j1_b255>
...
<action_j5_b0> ... <action_j5_b255>
```

After adding, call `model.resize_token_embeddings(len(tokenizer))`.

### Encode/decode API

Existing `scripts/tokenize_actions.py` `ActionTokenizer` class handles this. For the unified policy the caller is responsible for:
1. Splitting the 12-dim action into left `[0:6]` and right `[6:12]`
2. Encoding each half with the same `ActionTokenizer`
3. Concatenating the token lists: `[left_tokens(6), right_tokens(6)]` per timestep

> **Ambiguity resolved:** The existing `ActionTokenizer.encode_action()` takes a 6-dim vector, not 12. Rather than changing the tokenizer internals, the dataloader will call it twice per timestep (once per arm) and concatenate. This keeps the tokenizer simple and avoids breaking existing validation.

---

## Episode Augmentation

**Deliverable:** `scripts/augment_episodes.py` (modify existing)

### Changes from Siamese version

| Aspect | Siamese (old) | Unified (new) |
|--------|--------------|---------------|
| Arm-specific prompts | Separate left/right system + user prompts | Single unified prompt |
| Prompt content | "You are the LEFT/RIGHT arm..." | "You control both arms of a bimanual robot..." |
| Action history format | 6 tokens per timestep (one arm) | 12 tokens per timestep (both arms, left then right) |
| Output token count | 48 (8 x 6) | 96 (8 x 12) |

### Output JSON schema (per episode)

```json
{
  "episode_index": 0,
  "task": "grabber;picker;black_marker",
  "task_description": "Hand off the black marker (grabber to picker)",
  "num_frames_30fps": 1016,
  "num_frames_4fps": 136,
  "system_prompt": "You control both arms of a bimanual robot system. You receive three camera views (left wrist, right wrist, overhead) and recent joint history for all 12 joints. Output the next 8 timesteps of joint commands for both arms as 96 action tokens.",
  "user_prompt_template": "Task: {task_description}.\nHere are your three camera views and recent joint history (12 joints: left arm then right arm):\n{action_history}\nOutput the next 8 timesteps of joint commands (96 tokens: 8 steps x 12 joints).",
  "chain_of_thought_template": "<think>\n</think>",
  "frame_indices_4fps": [0, 8, 15, 23, ...]
}
```

---

## Training Dataloader

**Deliverable:** `scripts/dataloader.py` (rewrite existing `SiameseVLADataset`)

### Class: `BimanualVLADataset`

```python
class BimanualVLADataset(torch.utils.data.Dataset):
    """
    Each item is a TRL-compatible chat message list with 3 images,
    action history text, and 96 action tokens as the training target.

    For E episodes, each with F_i valid frames at 4 FPS:
        Total samples = sum(F_i for i in episodes)
        where F_i = num_frames_4fps_i - ACTION_CHUNK_LEN
    """
```

### Sample construction for a given `(episode, frame_idx_4fps)`

**1. Images (3x):**
- Left wrist: `observation.images.left.wrist_left` at frame `frame_indices_4fps[t]`
- Right wrist: `observation.images.right.wrist_right` at same frame
- Overhead: `observation.images.left.top` at same frame

> **Ambiguity resolved:** The overhead camera key is `observation.images.left.top` (it's associated with the "left" camera group in the LeRobot config, but it's a top-down view). We use it as-is.

All frames decoded via LeRobot's `decode_video_frames` with `torchcodec` backend. The Qwen3-VL processor handles resizing internally.

**2. Action history (text, 12 joints per timestep):**

At frame `t`, history is from 4 FPS indices `t-4, t-3, t-2, t-1` (clamped to 0 at episode start).

Each timestep's 12-dim action is encoded as 12 tokens: left arm 6 joints then right arm 6 joints, separated by `|`:
```
t-4: <action_j0_b42> <action_j1_b128> <action_j2_b64> <action_j3_b200> <action_j4_b15> <action_j5_b90> | <action_j0_b15> <action_j1_b88> <action_j2_b32> <action_j3_b150> <action_j4_b200> <action_j5_b42>
t-3: ...
t-2: ...
t-1: ...
```

> **Ambiguity resolved:** The `|` separator between left and right arm tokens in the history is a text convention for readability. In the output (training target), there is no separator — just 12 tokens per timestep in sequence. The model learns the `[left(6), right(6)]` ordering from the data.

**3. Action chunk (training target, 96 tokens):**
- 8 consecutive timesteps at 4 FPS starting from current frame
- Full 12-dim action vector per timestep
- Encoded as: `encode_action(action[0:6]) + encode_action(action[6:12])` per timestep
- Flattened to 96 token IDs total

**4. Chat message format:**
```python
messages = [
    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    {"role": "user", "content": [
        {"type": "image", "image": left_wrist_frame},
        {"type": "image", "image": right_wrist_frame},
        {"type": "image", "image": overhead_frame},
        {"type": "text", "text": user_prompt_with_action_history},
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": "<think>\n</think>" + action_chunk_tokens_text},
    ]},
]
```

> **Ambiguity resolved:** Qwen3-VL supports multiple images in a single user message. The processor tags them as `<|image_pad|>` tokens in sequence. We always order them: left wrist, right wrist, overhead — and the text prompt references them in this order so the model can learn the association. Verified this is supported by looking at the Qwen3-VL processor `apply_chat_template` behavior from the example code.

### Index table

No more `(episode, frame, arm)` triple — just `(episode, frame)` pairs:
```python
# Valid frames: need ACTION_CHUNK_LEN future steps
for ep_idx, ep_data in episodes.items():
    for t in range(ep_data["num_frames_4fps"] - ACTION_CHUNK_LEN):
        self._index_table.append((ep_idx, t))
```

### Dataset size estimate

11 episodes, average ~1,016 frames at 30 FPS -> ~136 frames at 4 FPS -> ~128 valid frames per episode (minus 8 for the chunk). Total: **~1,408 samples** (compared to ~2,816 with Siamese doubling).

> **Ambiguity resolved:** This is a small dataset. With 11 episodes we'll likely need multiple epochs and should watch for overfitting. The plan's suggestion of `max_steps=100` with effective batch size 16 would only see ~1,600 samples (barely 1 epoch). We should plan for more — see training config below.

---

## SFT Training Script

**Deliverable:** `scripts/train.py` (modify existing Siamese references)

Based on `models/example_sft.py`.

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | `nvidia/Cosmos-Reason2-2B` | Qwen3-VL architecture, robotics-aligned pretraining |
| Quantization | 4-bit NF4 (QLoRA) | Fit on single GPU |
| LoRA rank | 32 | Match example; good capacity vs. memory tradeoff |
| LoRA alpha | 32 | 1:1 with rank (from example) |
| LoRA targets | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Standard for Qwen3-VL |
| Learning rate | 2e-4 | Match example |
| Optimizer | `adamw_8bit` | Memory efficient |
| Batch size | 1 per device, gradient accumulation 8 -> effective 8 | 3 images per sample increases memory; may need batch=1 |
| `max_length` | `None` | Avoid truncating image/action tokens |
| Warmup steps | 5 | Match example |
| Logging | tensorboard, every step | |
| Max steps | Start with 10 for validation, then scale to full training | |

> **Ambiguity resolved:** The original plan had `per_device_train_batch_size=2`. With 3 images per sample (instead of 1), memory usage will be significantly higher. We default to batch size 1 and rely on gradient accumulation. This needs empirical testing — if OOM, we can reduce image resolution via the processor's `min_vision_tokens`/`max_vision_tokens` settings (as shown in `models/example_inference.py`).

### Steps

1. Load base model with `BitsAndBytesConfig` (4-bit NF4, fp16 compute, double quant)
2. Load processor with `Qwen3VLProcessor.from_pretrained()`
3. Create `ActionTokenizer` from dataset stats, register tokens with processor's tokenizer
4. Resize model embeddings: `model.resize_token_embeddings(len(processor.tokenizer))`
5. Configure LoRA adapter via `LoraConfig`
6. Initialize `BimanualVLADataset`
7. Create custom collate function that calls `processor.apply_chat_template()` to tokenize each sample's chat messages into input_ids, attention_mask, and labels
8. Configure `SFTTrainer` with training args
9. Train
10. Save adapter + tokenizer to `outputs/`

### Vision token budget

Each 1080x1920 image at default settings generates many vision tokens. With 3 images per sample this could be very large. We should limit via:
```python
processor.image_processor.size = {
    "shortest_edge": 256 * (32**2),    # 256 vision tokens min
    "longest_edge": 1024 * (32**2),    # 1024 vision tokens max
}
```

This keeps each image to 256-1024 vision tokens, so 3 images = 768-3072 vision tokens total. Tune based on GPU memory.

---

## Inference Script

**Deliverable:** `scripts/inference.py` (rewrite existing)

### Flow

1. Load base model + LoRA adapter via `PeftModel.from_pretrained()`
2. Load processor, create `ActionTokenizer`, register tokens (must match training)
3. Resize model embeddings
4. Accept input: 3 camera images + last 4 action vectors (12 floats each) + task prompt
5. Construct chat messages with system prompt + 3 images + action history text
6. `processor.apply_chat_template()` -> `model.generate(max_new_tokens=150)`
7. Parse output: extract `<think>...</think>`, then find 96 action token names
8. Decode: split 96 tokens into 8 groups of 12, split each group into left (first 6) and right (last 6), decode each half with `ActionTokenizer.decode_action()`, concatenate to get `[8, 12]` array
9. Return joint command sequence

> **Ambiguity resolved:** `max_new_tokens=150` gives headroom for the `<think></think>` tags (~5 tokens) + 96 action tokens + any tokenizer overhead. The original plan used 128 which might be tight.

### Real-time control loop (future)

```
while task_active:
    capture 3 images
    run inference -> [8, 12] chunk
    execute chunk at 4 Hz (send one [12] command every 250ms)
    overlap: start next inference while executing
```

---

## Task Dependency Graph

```
Task 1 (Tokenization)  ─────────────┐
                                     ├──> Task 3 (Dataloader) ──> Task 4 (Training) ──> Task 5 (Inference)
Task 2 (Augmentation)  ─────────────┘
```

Tasks 1 and 2 are independent. Task 3 depends on both. Task 4 depends on 3. Task 5 depends on 4 + 1.

### Implementation order

- **Task 1 — Action Tokenization:** Already implemented (`scripts/tokenize_actions.py`). No changes needed — the 6-joint API works for unified policy by calling it twice per timestep.
- **Task 2 — Episode Augmentation:** Needs rewrite to remove arm-specific prompts and use unified prompt templates.
- **Task 3 — Dataloader:** Needs rewrite from `SiameseVLADataset` to `BimanualVLADataset`. Major changes: 3 images, 12-joint history, 96 output tokens, no arm index.
- **Task 4 — Training Script:** Needs writing. Closely follows `models/example_sft.py` with custom dataset and vision token budget tuning.
- **Task 5 — Inference Script:** Needs writing. Based on `models/example_inference.py` with action token parsing added.

---

## Success Criteria

| # | Criterion | How to verify |
|---|-----------|---------------|
| 1 | **Data loading works.** Dataset yields `(3 images, action_history, prompt, 96 action tokens)` at 4 FPS. | Iterate one batch, assert shapes and token validity. |
| 2 | **Dataset size is correct.** Total samples = `sum(valid_frames per episode)`. | Assert `len(dataset) == sum(num_frames_4fps_i - 8 for each episode)`. |
| 3 | **Augmentation generates unified prompts.** No arm-specific prompts. | Inspect JSON files. |
| 4 | **Action discretization is invertible.** Round-trip error < 1 bin width. | Already validated by existing `tokenize_actions.py`. |
| 5 | **SFT training runs.** 10 gradient steps complete without OOM or NaN. | Training loss decreases. |
| 6 | **Inference works.** Sample input produces `[8, 12]` array of valid joint values. | Script outputs array from a training episode sample. |

---

## Ambiguities Resolved

1. **Dataset mismatch.** plan.md references `20260228_150311` (24 episodes). Disk has `20260226_211245` (11 episodes). Used what exists. Scripts are path-parameterized.

2. **Token vocabulary for 12 joints.** Chose to keep 6 token types (1,536 tokens) rather than 12 (3,072). Both arms are identical SO-101s sharing the same joint ranges. Left vs. right is distinguished by position in the output sequence, not by token identity. This halves vocabulary overhead.

3. **Overhead camera inclusion.** plan.md explicitly said "ignored for training." We now include it — it provides the global workspace view critical for coordination that the wrist cameras alone can't give.

4. **Action history uses `action` not `observation.state`.** Both are `[12]` float arrays in the dataset. The plan was ambiguous. We use `action` (commanded positions) for consistency — the model predicts future actions, so conditioning on past actions is more coherent than conditioning on observed state (which may lag due to servo dynamics).

5. **Batch size reduction.** 3 images per sample (vs 1 in Siamese) roughly triples vision token count. Reduced `per_device_train_batch_size` from 2 to 1 to avoid OOM. Effective batch size is still reasonable via gradient accumulation.

6. **`|` separator in action history.** Added a pipe separator between left and right arm tokens in the history text for readability. The output target has no separator — just 12 tokens per timestep in sequence. This is a formatting choice; the model will learn either convention.

7. **Small dataset concern.** ~1,408 samples is small. Plan's `max_steps=100` barely covers 1 epoch at effective batch 8. Full training will need multiple epochs and possibly augmentation (e.g., image jitter, action noise). Flagging this as a risk but not solving it in V1.
