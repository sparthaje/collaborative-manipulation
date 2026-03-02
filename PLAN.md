# PLAN.md — Siamese VLA Fine-Tuning for Bimanual Object Handoff

## Section 1: Goals

### Objective

Fine-tune **Cosmos-Reason2-2B** (Qwen3-VL architecture) with QLoRA so that a single policy checkpoint can control either arm of a bimanual SO-101 robot during object handoff. At inference time each arm runs its own copy of the model, receiving its current wrist camera image, recent action history (1 s at 4 FPS), and a text prompt identifying which arm it is. The model outputs a 2-second **action chunk** — 8 timesteps of discretized 6-DoF joint actions as tokens (48 tokens total).

### Success Criteria

| # | Criterion | How to verify |
|---|-----------|---------------|
| 1 | **Data loading works end-to-end.** A custom PyTorch `Dataset` yields `(image, action_history, prompt, action_chunk_tokens)` tuples at 4 FPS. Each sample has 1 current wrist image, 4 prior actions (1 s history), and an 8-step action chunk (2 s future, 48 tokens). | Unit test: iterate one batch, assert tensor shapes and token validity. |
| 2 | **Siamese pairing doubles effective data.** Every valid timestep produces two training samples — one from the left-wrist POV (with left-arm prompt + left-arm action chunk) and one from the right-wrist POV (with right-arm prompt + right-arm action chunk). | Assert `len(dataset) == 2 * num_valid_frames`. |
| 3 | **Augmentation metadata is generated.** A standalone script reads episode parquet + info.json and writes per-episode JSON files containing left/right prompts and chain-of-thought annotations. | Spot-check a few JSON files against expected format. |
| 4 | **Action discretization is invertible.** 6 continuous joint values → 6 token IDs → 6 reconstructed values, with reconstruction error < 1 bin width. | Round-trip test on dataset min/max range. |
| 5 | **SFT training loop runs.** TRL `SFTTrainer` with QLoRA adapter completes at least 10 gradient steps without OOM or NaN on a single GPU. | Training loss decreases over 10 steps. |
| 6 | **Inference pipeline works.** Load base model + LoRA adapter, feed a wrist image + action history + prompt, decode generated action chunk tokens back to continuous joint commands. | Script produces an 8×6 array of joint values from a sample episode. |

---

## Section 2: Tasks

### Task 0 — Understand the Existing Dataset

**Status:** ✅ Done (summarized below for reference)

**Key findings from analysis:**

- **Dataset:** `data/grabber_picker_black_marker_20260228_150311`
- **Episodes:** 24, totaling 25,724 frames at 30 FPS
- **Action/State shape:** `[12]` — 6 joints per arm concatenated as `[left(6), right(6)]`
- **Joint names (per arm):** `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos`
- **Video keys:**
  - `observation.images.left.wrist_left` — left-arm egocentric (1080×1920, AV1, 30 FPS)
  - `observation.images.right.wrist_right` — right-arm egocentric (1080×1920, AV1, 30 FPS)
  - `observation.images.left.top` — overhead (ignored for training)
- **Video storage:** chunked MP4 files, multiple episodes per file, with `from_timestamp`/`to_timestamp` offsets per episode in episodes parquet
- **LeRobot dataset class:** `lerobot.datasets.lerobot_dataset.LeRobotDataset`
  - Supports `delta_timestamps` for history frame loading (clamps to episode boundaries, provides `_is_pad` flags)
  - Video frames decoded via `torchcodec` or `pyav` backend, returned as `float32 [0,1]` tensors
  - No built-in FPS resampling — frame selection is index-based
- **Reference model:** Cosmos-Reason2-2B (Qwen3-VL), SFT with TRL + QLoRA, `Qwen3VLForConditionalGeneration`
- **Inference pattern:** `processor.apply_chat_template()` → `model.generate()` → `processor.batch_decode()`, supports image and video inputs

---

### Task 1 — Action Tokenization Scheme

**Deliverable:** `scripts/tokenize_actions.py` — a utility module (importable, not just CLI) that defines:

1. **Bin configuration:**
   - Compute per-joint min/max from `meta/stats.json` (use `q01`/`q99` with a small margin to handle outliers).
   - Divide each joint's range into `N` uniform bins (start with `N=256`).
   - Each of the 6 joints shares the same `N`-width token block but has its own offset: joint `j` uses token IDs `[reserved_start + j*N, reserved_start + (j+1)*N)`.

2. **Token reservation in Qwen3-VL tokenizer:**
   - Add `6 * N` special tokens to the tokenizer via `tokenizer.add_tokens([...])`.
   - Naming convention: `<action_j0_b0>` … `<action_j0_b255>`, `<action_j1_b0>` … etc.
   - Resize model embeddings after adding tokens: `model.resize_token_embeddings(len(tokenizer))`.

3. **Encode / decode functions:**
   - `encode_action(joint_values: np.ndarray[6]) -> list[int]` — single-timestep continuous → 6 token IDs.
   - `decode_action(token_ids: list[int]) -> np.ndarray[6]` — single-timestep 6 token IDs → bin-center values.
   - `encode_action_chunk(chunk: np.ndarray[T, 6]) -> list[int]` — multi-timestep continuous → flat token ID list (T×6 tokens).
   - `decode_action_chunk(token_ids: list[int], num_joints: int = 6) -> np.ndarray[T, 6]` — flat token IDs → multi-timestep bin-center values.

4. **Round-trip validation:** assert max reconstruction error < 1 bin width over the full dataset.

**Key design decision — joint quantization:**
All 12 action dimensions are in **degrees** (6 per arm: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper). Both arms are SO-101s with the same joint layout, so joint index `i` on the left arm and joint index `i` on the right arm represent the same physical degree-of-freedom in the same unit. This means we can define a **single set of 6 bin ranges** (one per joint index) that covers both arms. For each training sample we use only the 6 dims corresponding to the arm in the prompt (left sample → `action[0:6]`, right sample → `action[6:12]`), and the same tokenizer maps them identically. Use the component-wise min of both arms' `q01` and max of both arms' `q99` to define these shared per-joint ranges.

**Dependencies:** None.

**Validation:** Run `encode_action(decode_action(tokens)) == tokens` on 1000 random samples.

---

### Task 2 — Episode Augmentation Script

**Deliverable:** `scripts/augment_episodes.py`

**Inputs:** dataset root path (reads `meta/info.json`, `meta/episodes/.../file-000.parquet`, `data/.../file-000.parquet`).

**Outputs:** `<dataset_root>/augmentation/episode_<idx>.json` per episode, containing:

```json
{
  "episode_index": 3,
  "task": "grabber;picker;black_marker",
  "num_frames_30fps": 1050,
  "num_frames_4fps": 140,
  "left_arm": {
    "system_prompt": "You are controlling the LEFT arm of a two-arm robot system. ...",
    "user_prompt_template": "You are the LEFT arm. The task is: {task_description}. Here is your current wrist camera image and your last 4 actions:\n{action_history}\nOutput the next 8 joint actions (2 seconds at 4 FPS) as 48 tokens.",
    "chain_of_thought_template": "I am the left arm. I can see {object} in my gripper/approaching. I need to {action_description} to complete the handoff. My next joint commands are: {action_chunk_tokens}"
  },
  "right_arm": {
    "system_prompt": "You are controlling the RIGHT arm of a two-arm robot system. ...",
    "user_prompt_template": "You are the RIGHT arm. The task is: {task_description}. Here is your current wrist camera image and your last 4 actions:\n{action_history}\nOutput the next 8 joint actions (2 seconds at 4 FPS) as 48 tokens.",
    "chain_of_thought_template": "I am the right arm. I can see {object} approaching/in position. I need to {action_description} to complete the handoff. My next joint commands are: {action_chunk_tokens}"
  },
  "frame_indices_4fps": [0, 7, 15, 22, ...]cccccdbcfrcfucdivgbfhcctlikklknhhvrilvntnbhh

  
}
```

**Logic:**

1. Parse the task name to determine handoff direction and object.
   - `"grabber;picker;black_marker"` → left arm is grabber/picker, right arm is grabber/picker (roles may vary; encode both).
2. Compute the 4 FPS frame indices: `range(0, num_frames, 30 // 4)` → every 7th or 8th frame (use `round(i * 30 / 4)` for each `i` in `range(num_frames_4fps)`).
3. Generate prompt templates. The actual action tokens are filled in at training time by the dataloader, not baked into the JSON.
4. Chain-of-thought is a template string. For V1 just write <think></think>

**Dependencies:** None.

**Validation:** Assert all referenced frame indices are within episode bounds. Assert JSON schema is consistent across episodes.

---

### Task 3 — Training Dataloader

**Deliverable:** `scripts/dataloader.py` — a PyTorch `Dataset` class.

**Design:**

```
class SiameseVLADataset(torch.utils.data.Dataset):
    """
    Each item is one (arm_pov, prompt, action_tokens) tuple.

    For E episodes, each with F resampled frames:
      - Total samples = 2 * sum(F_i for i in episodes)
      - Sample index maps to (episode, frame, arm) via precomputed index table.
    """
```

**Sample construction for a given `(episode, frame_idx_4fps, arm)`:**

Valid sample indices: `frame_idx_4fps` must allow an 8-step action chunk, so `frame_idx_4fps` ∈ `[0, num_frames_4fps - 8)`. Each valid index produces 2 samples (left + right).

1. **Current image:** Extract the single frame at `frame_idx_4fps` (mapped to 30 FPS index).
   - Source: `observation.images.left.wrist_left` or `observation.images.right.wrist_right` depending on `arm`.
   - Frame extraction from MP4: use LeRobot video utilities (`decode_video_frames` with `torchcodec` backend) by computing the timestamp from the episode's `from_timestamp` + frame offset.
   - Resize handled by Qwen3-VL processor internally.

2. **Action history (text input):** The last 4 actions at 4 FPS (1 second of history) for the selected arm.
   - At frame `t`, history actions are from 4 FPS indices `t-4, t-3, t-2, t-1`.
   - Handle episode start: clamp to first frame if fewer than 4 prior frames exist.
   - Each historical action is the 6-joint vector for the selected arm, encoded as action tokens.
   - Formatted as a text block in the user prompt, e.g.:
     ```
     t-4: <action_j0_b42> <action_j1_b128> <action_j2_b64> <action_j3_b200> <action_j4_b15> <action_j5_b90>
     t-3: ...
     t-2: ...
     t-1: ...
     ```

3. **Prompt:** Load from `augmentation/episode_<idx>.json`, select `left_arm` or `right_arm` based on `arm`. Fill in `{action_history}` with the formatted history from step 2.

4. **Action chunk tokens (training target):**
   - Read actions from the parquet at the 8 consecutive 30 FPS frame indices starting at the current 4 FPS index (i.e., frames `t, t+1, ..., t+7` in 4 FPS indexing, each mapped to 30 FPS).
   - Slice to `[0:6]` for left arm or `[6:12]` for right arm.
   - Encode all 8 timesteps via `encode_action_chunk()` from Task 1 → 48 token IDs.

5. **Format as TRL-compatible chat messages:**
   ```python
   messages = [
       {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
       {"role": "user", "content": [
           {"type": "image", "image": current_frame},  # single image
           {"type": "text", "text": user_prompt_with_action_history},
       ]},
       {"role": "assistant", "content": [
           {"type": "text", "text": chain_of_thought_and_action_chunk},
       ]},
   ]
   ```
   The assistant response (the training target) is the chain-of-thought followed by the 48 action tokens (8 timesteps × 6 joints). For example:
   ```
   <think></think><action_j0_b42> <action_j1_b128> <action_j2_b64> <action_j3_b200> <action_j4_b15> <action_j5_b90> <action_j0_b43> <action_j1_b127> ... (48 tokens total)
   ```
   The model learns to produce the `<think>...</think>` block and then the 48 action tokens (8 groups of 6, one per future timestep). At inference time the full assistant output is parsed: the action tokens are extracted and decoded back to an 8×6 array of joint angles.

6. **Collation:** Use `processor.apply_chat_template()` inside a custom collate function or pre-tokenize.

**Dependencies:** Task 1 (action tokenization), Task 2 (augmentation JSON).

**Validation:**
- `len(dataset)` matches `2 * total_valid_frames` (where valid frames exclude the last 7 per episode to allow full 8-step chunks).
- Iterate 10 samples, verify image tensor shape is `(C, H, W)` with values in `[0, 1]`.
- Verify each sample has 48 action tokens (8 timesteps × 6 joints) that are valid token IDs in the extended tokenizer.
- Verify action history is correctly clamped at episode boundaries.
- Verify no cross-episode action chunk leakage.

---

### Task 4 — SFT Training Script

**Deliverable:** `scripts/train.py`

**Based on:** `models/example_sft.py` (TRL + QLoRA pattern).

**Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | `nvidia/Cosmos-Reason2-2B` | Qwen3-VL architecture, robotics-aligned pretraining |
| Quantization | 4-bit NF4 (QLoRA) | Fit on single GPU |
| LoRA rank | 32 | Match example; good balance of capacity vs. memory |
| LoRA targets | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Standard for Qwen3-VL |
| Learning rate | 2e-4 | Match example |
| Batch size | 2 per device, gradient accumulation 8 → effective 16 | Adjust based on GPU memory |
| Max steps | Start with 100, then scale | Validate training loop first |
| Input | Single image + action history text | No video processing needed; action context is textual |
| `max_length` | `None` | Avoid truncating image/action tokens |

**Steps:**
1. Load base model with `BitsAndBytesConfig`.
2. Add action tokens to tokenizer (from Task 1), resize embeddings.
3. Configure LoRA adapter.
4. Initialize `SiameseVLADataset` (from Task 3).
5. Configure `SFTTrainer` with custom data collator that uses `processor.apply_chat_template()`.
6. Train.
7. Save adapter to `outputs/`.

**Dependencies:** Task 1, Task 2, Task 3.

**Validation:**
- Training loss decreases over first 10 steps.
- No OOM errors.
- Saved adapter can be loaded with `PeftModel.from_pretrained()`.

---

### Task 5 — Inference Script

**Deliverable:** `scripts/inference.py`

**Based on:** `models/example_inference.py`.

**Flow:**
1. Load base model + LoRA adapter.
2. Add action tokens to tokenizer (must match training), resize embeddings.
3. Accept input: a single wrist camera image + last 4 actions (1 s history at 4 FPS) + arm identifier (`left` or `right`).
4. Construct chat messages with system prompt + image + action history text + user prompt.
5. `processor.apply_chat_template()` → `model.generate(max_new_tokens=128)`.
6. Parse output text: extract 48 action token names → `decode_action_chunk()` → 8×6 array of joint commands.
7. Print / return the joint command sequence.

**Inference-time contract:**
- Input: 1 RGB image from one wrist camera + 4 prior action vectors (6 floats each) + `"left"` or `"right"`.
- Output: 8×6 float array (degrees) — 2 seconds of joint commands at 4 FPS for that arm.

**Dependencies:** Task 1 (tokenization), Task 4 (trained adapter).

**Validation:**
- Feed an image + action history from a training episode, verify output is an 8×6 array of valid floats within joint range.
- Feed left-arm and right-arm inputs from the same timestep, verify they produce different (appropriate) action sequences.

---

### Task Dependency Graph

```
Task 1 (Tokenization)  ─────────────┐
                                     ├──→ Task 3 (Dataloader) ──→ Task 4 (Training) ──→ Task 5 (Inference)
Task 2 (Augmentation)  ─────────────┘
```

Tasks 1 and 2 are independent and can be developed in parallel. Task 3 depends on both. Task 4 depends on Task 3. Task 5 depends on Task 4 (for the adapter weights) and Task 1 (for decoding).
