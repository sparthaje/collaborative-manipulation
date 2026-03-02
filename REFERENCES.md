# REFERENCES.md — Implementation Resources for Siamese VLA Fine-Tuning

Quick-reference for every task in [PLAN.md](./PLAN.md).

---

## Installed Package Versions

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.7.1+cu126 | CUDA 12.6 |
| lerobot | 0.4.4 | Dataset loading & video decoding |
| torchcodec | 0.5 | Default video backend |
| trl | 0.29.0 | SFT trainer |
| peft | 0.18.1 | LoRA / QLoRA adapters |
| bitsandbytes | 0.49.2 | 4-bit quantization |
| transformers | 4.57.6 | Qwen3VL model & processor |

---

## 1. Dataset On-Disk Structure

**Root:** `data/grabber_picker_black_marker_20260228_150311/`

```
meta/
  info.json              # fps=30, 24 episodes, 25724 frames, feature schemas
  stats.json             # per-feature: min/max/mean/std/q01/q99 (12-dim actions)
  tasks.parquet          # single task: "grabber;picker;black_marker"
  episodes/
    chunk-000/
      file-000.parquet   # 24 rows — episode boundaries, video timestamps, per-ep stats
data/
  chunk-000/
    file-000.parquet     # 25724 rows — columns: action[12], observation.state[12],
                         #   timestamp, frame_index, episode_index, index, task_index
videos/
  observation.images.left.wrist_left/chunk-000/   # 4 MP4s, ~630 MB total
  observation.images.right.wrist_right/chunk-000/  # 3 MP4s, ~477 MB total
  observation.images.left.top/chunk-000/           # 4 MP4s, ~701 MB (unused for training)
```

### Action Stats (from `meta/stats.json`)

Dimensions 0–5 = left arm, 6–11 = right arm. All values in degrees.

| Idx | Joint | q01 (L) | q99 (L) | q01 (R) | q99 (R) |
|-----|-------|---------|---------|---------|---------|
| 0/6 | shoulder_pan | -8.93 | 69.93 | -67.50 | 9.50 |
| 1/7 | shoulder_lift | -103.16 | 1.81 | -112.14 | 40.17 |
| 2/8 | elbow_flex | -22.07 | 97.19 | -64.20 | 97.27 |
| 3/9 | wrist_flex | -100.28 | 89.86 | -103.63 | 104.24 |
| 4/10 | wrist_roll | -160.13 | -53.65 | -169.89 | -91.70 |
| 5/11 | gripper | 1.29 | 25.62 | 1.38 | 27.80 |

Per PLAN.md Task 1: use component-wise `min(q01_left, q01_right)` and `max(q99_left, q99_right)` to define shared 6-joint bin ranges.

### Episodes Parquet Key Columns

```
episode_index          — int64, 0–23
tasks                  — list<string>, e.g. ['grabber;picker;black_marker']
length                 — int64, frames per episode (e.g. 1313)
dataset_from_index     — int64, global start index
dataset_to_index       — int64, global end index (exclusive)
videos/{cam}/from_timestamp  — float64, start time in MP4
videos/{cam}/to_timestamp    — float64, end time in MP4
videos/{cam}/chunk_index     — int64
videos/{cam}/file_index      — int64
```

---

## 2. LeRobot Python API

**Package path:** `.venv/lib/python3.10/site-packages/lerobot/`

### LeRobotDataset (lerobot.datasets.lerobot_dataset)

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id="local_dataset",
    root="data/grabber_picker_black_marker_20260228_150311",
    episodes=[0, 1, 2],               # optional subset
    delta_timestamps={                 # history frames as time deltas (seconds)
        "observation.images.left.wrist_left": [-1.0, -0.75, -0.5, -0.25, 0.0],
    },
    tolerance_s=1e-4,
    video_backend="torchcodec",        # or "pyav"
)

sample = ds[0]  # dict with all features + decoded video tensors
```

**Key properties:** `ds.fps`, `ds.num_frames`, `ds.num_episodes`, `ds.features`, `ds.meta.stats`

**`__getitem__` flow:**
1. Load row from data parquet (action, state, timestamps)
2. If `delta_timestamps` set → `_get_query_indices()` computes multi-step indices, clamps to episode boundaries
3. `_query_videos()` → shifts timestamps by episode `from_timestamp`, calls `decode_video_frames()`
4. Returns dict with tensors

### Video Decoding (lerobot.datasets.video_utils)

```python
from lerobot.datasets.video_utils import decode_video_frames

# Returns float32 tensor [num_timestamps, H, W, 3] in [0, 1]
frames = decode_video_frames(
    video_path="videos/.../file-000.mp4",
    timestamps=[0.0, 0.25, 0.5, 0.75, 1.0],  # absolute timestamps in the MP4
    tolerance_s=1e-4,
    backend="torchcodec",  # or "pyav"
)
```

**Important:** timestamps are absolute within the MP4 file. For a given episode, offset by `from_timestamp` from the episodes parquet:
```python
ep_meta = ds.meta.episodes[ep_idx]
from_ts = ep_meta["videos/observation.images.left.wrist_left/from_timestamp"]
absolute_ts = from_ts + relative_ts_within_episode
```

**VideoDecoderCache:** Global singleton caches open decoders for efficiency. Available as `lerobot.datasets.video_utils._default_decoder_cache`.

### Stats Loading (lerobot.datasets.utils)

```python
from lerobot.datasets.utils import load_stats, load_info, load_episodes

stats = load_stats(Path("data/grabber_picker_black_marker_20260228_150311"))
# stats["action"]["q01"]  → np.ndarray shape (12,)
# stats["action"]["q99"]  → np.ndarray shape (12,)
# stats["action"]["min"]  → np.ndarray shape (12,)
# stats["action"]["max"]  → np.ndarray shape (12,)
```

### File Path Templates (lerobot.datasets.utils)

```python
DEFAULT_DATA_PATH     = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH    = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
```

---

## 3. Cosmos-Reason2 / Qwen3-VL Model API

**Model:** `nvidia/Cosmos-Reason2-2B` — Qwen3-VL architecture fine-tuned for robotics reasoning.

### Loading (from models/example_inference.py)

```python
import transformers

model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
    "nvidia/Cosmos-Reason2-2B",
    dtype=torch.float16,        # or "auto"
    device_map="auto",
    attn_implementation="sdpa",
)
processor = transformers.Qwen3VLProcessor.from_pretrained("nvidia/Cosmos-Reason2-2B")
```

### Vision Token Budget

```python
PIXELS_PER_TOKEN = 32**2  # 1024 pixels per token

processor.image_processor.size = {
    "shortest_edge": 256 * PIXELS_PER_TOKEN,    # min 256 tokens
    "longest_edge": 8192 * PIXELS_PER_TOKEN,    # max 8192 tokens
}
processor.video_processor.size = { ... }  # same structure
```

### Chat Template — Image Input (with Action History in Text)

```python
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are controlling the LEFT arm..."}]},
    {"role": "user", "content": [
        {"type": "image", "image": current_frame},  # single image, MUST come before text
        {"type": "text", "text": "... your last 4 actions:\nt-4: <action_j0_b42> ...\nt-3: ...\nt-2: ...\nt-1: ...\nOutput the next 8 joint actions (2 seconds at 4 FPS) as 48 tokens."},
    ]},
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
```

**Image input formats:** file path string, URL, PIL Image, or torch tensor.

### Generation & Decoding

```python
generated_ids = model.generate(**inputs.to(model.device), max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
```

### Adding Custom Tokens (for action tokenization)

```python
new_tokens = [f"<action_j{j}_b{b}>" for j in range(6) for b in range(256)]
num_added = processor.tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(processor.tokenizer))
```

---

## 4. QLoRA + SFT Training API

**Source:** `models/example_sft.py`

### Quantization Config

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "nvidia/Cosmos-Reason2-2B",
    dtype="auto",
    device_map="auto",
    quantization_config=bnb_config,
)
```

### LoRA Config

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)
```

### SFT Training

```python
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    max_steps=10,                       # or num_train_epochs=1
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,      # effective batch = 16
    warmup_steps=5,
    learning_rate=2e-4,
    optim="adamw_8bit",
    max_length=None,                    # CRITICAL for VLMs — don't truncate image tokens
    output_dir="outputs/...",
    logging_steps=1,
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,              # must yield chat-format dicts
    peft_config=peft_config,
)
trainer.train()
trainer.save_model(output_dir)
```

### Loading Fine-Tuned Adapter

```python
from peft import PeftModel

model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)
```

---

## 5. Task-Specific Reference Map

### Task 1 — Action Tokenization

| What | Where |
|------|-------|
| Per-joint stats (q01/q99) | `meta/stats.json` → `stats["action"]["q01"][0:6]` and `[6:12]` |
| Adding special tokens | `processor.tokenizer.add_tokens(list_of_strings)` |
| Resizing embeddings | `model.resize_token_embeddings(len(tokenizer))` |
| Bin math | `bin_idx = int((value - low) / (high - low) * N)`, clamped to [0, N-1] |
| Decode back | `value = low + (bin_idx + 0.5) * (high - low) / N` (bin center) |
| Action chunk encode | `encode_action_chunk(np.ndarray[T,6]) -> list[int]` (T×6 flat token list) |
| Action chunk decode | `decode_action_chunk(list[int], num_joints=6) -> np.ndarray[T,6]` |

### Task 2 — Episode Augmentation

| What | Where |
|------|-------|
| Episode count & lengths | `meta/episodes/chunk-000/file-000.parquet` → `length` column |
| Task name | `meta/tasks.parquet` → "grabber;picker;black_marker" |
| 4 FPS resampling | `frame_idx_30fps = round(i * 30 / 4)` for i in range(num_frames_4fps) |
| num_frames_4fps | `ceil(num_frames_30fps * 4 / 30)` |

### Task 3 — Dataloader

| What | Where |
|------|-------|
| Single-frame decoding | `lerobot.datasets.video_utils.decode_video_frames()` (1 timestamp) |
| Episode video offsets | Episodes parquet `videos/{cam}/from_timestamp` |
| Action slicing | Left: `action[0:6]`, Right: `action[6:12]` |
| Action history | 4 prior actions at 4 FPS, encoded as action tokens in user prompt text |
| Action chunk (output) | 8 future timesteps at 4 FPS → 48 tokens (8×6 joints) |
| Valid frame range | `[0, num_frames_4fps - 8)` per episode (need room for 8-step chunk) |
| Chat message format | See Section 3 above (system/user/assistant with image + text) |
| Padding at episode start | Clamp action history indices to 0 if fewer than 4 prior frames |

### Task 4 — Training

| What | Where |
|------|-------|
| Full SFT example | `models/example_sft.py` |
| QLoRA config | See Section 4 above |
| SFTTrainer dataset format | List of chat message dicts (system/user/assistant) |
| `max_length=None` | Required for VLMs to avoid truncating image/action tokens |

### Task 5 — Inference

| What | Where |
|------|-------|
| Full inference example | `models/example_inference.py` |
| Image + action history input | Single image + 4 prior actions as tokenized text |
| Media before text | Required ordering in content arrays |
| Token parsing | Extract 48 `<action_j*_b*>` tokens from generated text, `decode_action_chunk()` → 8×6 array |
| `max_new_tokens=128` | Must accommodate `<think>` block + 48 action tokens |

---

## 6. Existing Scripts

| Script | Purpose |
|--------|---------|
| `scripts/record.py` | Bimanual teleoperation recording → LeRobot dataset |
| `scripts/merge_lerobot_bimanual.py` | Merge two single-arm datasets into bimanual |
| `scripts/visualize_lerobot_v3.py` | Qt GUI for browsing LeRobot v3 datasets |
| `scripts/find_camera.py` | Camera discovery utility |

---

## 7. Key Design Decisions & Gotchas

1. **Video timestamps are absolute within MP4 files.** Always add `from_timestamp` from the episodes parquet before calling `decode_video_frames()`.

2. **Multiple episodes per MP4.** Videos are chunked by file size (~200 MB), not by episode. A single MP4 may contain frames from multiple consecutive episodes.

3. **torchcodec is the preferred backend** (faster than pyav). Falls back to pyav if torchcodec unavailable.

4. **`max_length=None` is critical** in SFTConfig for VLMs — truncation can remove vision tokens and cause training crashes.

5. **Action dims are in degrees** (not radians). Both arms are SO-101 with identical joint layouts.

6. **Shared bin ranges across arms** — joint i on left and joint i on right use the same bin boundaries. Compute ranges as `min(q01_left[i], q01_right[i])` and `max(q99_left[i], q99_right[i])`.

7. **TRL SFTTrainer expects chat-format data** — each sample is a list of message dicts with `role` and `content` keys. For multimodal, content contains typed items (`{"type": "video", ...}`, `{"type": "text", ...}`).

8. **Single image, not video.** Each sample uses one current wrist camera frame (not a video clip). Action history is provided as tokenized text in the prompt instead. This reduces vision token count and simplifies the data pipeline.

9. **Action chunks — 2 seconds at 4 FPS.** The model outputs 48 action tokens per sample (8 timesteps × 6 joints). Valid training frames must have room for the full 8-step chunk within the episode. Action history (4 prior steps) is clamped at episode start.

10. **Image resolution:** Raw is 1080x1920 (portrait). The Qwen3-VL processor handles resizing internally based on the vision token budget.
