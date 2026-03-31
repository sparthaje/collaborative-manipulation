"""Flask-based GPU inference server for the Siamese VLA model.

Usage:
    python -m scripts.server --checkpoint path/to/adapter --data_dir data --port 5000
"""

from __future__ import annotations

import argparse
import io
import json
import re

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

from scripts.augment_episodes import parse_task_description
from scripts.inference import (
    SYSTEM_PROMPTS,
    USER_PROMPT_TEMPLATES,
    format_action_history,
    load_model,
    predict,
)

app = Flask(__name__)

# Global state populated at startup
model = None
processor = None
action_tokenizer = None


def extract_cot(raw_output: str) -> str:
    """Extract <think>...</think> block from raw model output."""
    match = re.search(r"<think>.*?</think>", raw_output, re.DOTALL)
    return match.group(0) if match else ""


@app.route("/health", methods=["POST"])
def health():
    device = str(next(model.parameters()).device) if model is not None else "unknown"
    return jsonify({
        "status": "ready" if model is not None else "not_loaded",
        "device": device,
    })


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    # Parse images
    left_file = request.files.get("left_image")
    right_file = request.files.get("right_image")
    if left_file is None or right_file is None:
        return jsonify({"error": "Both left_image and right_image are required"}), 400

    left_image = Image.open(io.BytesIO(left_file.read())).convert("RGB")
    right_image = Image.open(io.BytesIO(right_file.read())).convert("RGB")

    # Parse task
    task_str = request.form.get("task")
    if task_str is None:
        return jsonify({"error": "task is required"}), 400
    left_description, right_description = parse_task_description(task_str)

    # Parse history
    try:
        left_history = np.array(json.loads(request.form["left_history"]), dtype=np.float64)
        right_history = np.array(json.loads(request.form["right_history"]), dtype=np.float64)
    except (KeyError, json.JSONDecodeError, ValueError) as e:
        return jsonify({"error": f"Invalid history: {e}"}), 400

    if left_history.shape != (4, 6) or right_history.shape != (4, 6):
        return jsonify({"error": f"History must be shape (4,6), got left={left_history.shape} right={right_history.shape}"}), 400

    # Run inference for each arm sequentially
    arms_config = [
        ("left", left_image, left_history, left_description),
        ("right", right_image, right_history, right_description),
    ]

    actions = []
    cot_blocks = []

    for arm, image, history, task_description in arms_config:
        history_text = format_action_history(history, action_tokenizer)
        system_prompt = SYSTEM_PROMPTS[arm]
        user_prompt = USER_PROMPT_TEMPLATES[arm].format(
            task_description=task_description,
            action_history=history_text,
        )

        action_chunk, inference_time, raw_output = predict(
            model,
            processor,
            action_tokenizer,
            image,
            history,
            arm,
            task_description=task_description,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=128,
        )

        actions.append(action_chunk.tolist())
        cot_blocks.append(extract_cot(raw_output))

    return jsonify({
        "actions": actions,
        "cot": cot_blocks,
        "shape": [2, 8, 6],
    })


def main():
    global model, processor, action_tokenizer

    parser = argparse.ArgumentParser(description="Siamese VLA Inference Server")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Parent directory with datasets")
    parser.add_argument("--port", type=int, default=5000,
                        help="Server port")
    args = parser.parse_args()

    print("Loading model...")
    model, processor, action_tokenizer = load_model(args.checkpoint, args.data_dir)
    print("Model loaded. Starting server.")

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
