Create a `PLAN.md` document for this project.

The goal of this prompt is to generate a clear implementation plan, not code changes yet.

Project context:
- I have recorded demonstrations from a two-arm robot system performing a shared manipulation task.
- The task is object handoff between arms (handoff direction can vary ie pass from left arm to right or vice versa).
- I want to fine-tune a VLM-based policy using a Siamese setup so inference can run decentralized on both arms. That is to say the VLA policy can take in either the wrist camera feed from the left arm or right arm and output the correct action space, the prompt will tell the VLA which arm it is. 

What I want analyzed:
1. LeRobot dataloaders in the local pip environment (`.venv`) and how they currently work.
2. Dataset episode structure for `grabber_picker_black_marker_20260228_150311`.
3. `scripts/visualize_lerobot_v3.py` to understand what fields/signals are available. Note, there is a wrist video for both arms and a top video with a separately mounted camera. The top video can be ignored. 
4. Reference training/inference patterns in:
   - `models/example_sft.py`
   - `models/example_inference.py`

Planned data/training requirements:
- Resample demonstrations from 30 FPS to 4 FPS.
- Use the last 1 second of history as model context.
- Train using both arm POV streams per frame (Siamese-style pairing).
- Augment each episode with (this should be its own script where it takes in the episode metadata + action data and outputs this information):
  - left-arm prompt
  - right-arm prompt
  - expected chain-of-thought output
- This augmentation can be stored as JSON in the dataset directory. Whatever works best.
- Action space: 6D per step (degrees); discretize each dimension into `N` bins/tokens using a shared reserved token space from the Qwen3VL tokenizer.

Output format for `PLAN.md` (required):
- Section 1: **Goals**
  - Define the objective and success criteria for data understanding, augmentation, and training readiness.
- Section 2: **Tasks**
  - Provide an ordered task list with concrete deliverables, dependencies, and validation checks.

When generating `PLAN.md`, keep it practical and execution-oriented so it can be implemented directly.
