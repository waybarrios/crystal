#!/usr/bin/env python3
"""
Model-agnostic CRYSTAL inference.

Runs any vision-language model over the CRYSTAL benchmark and writes per-sample
predictions in the ``{"reasoning_steps": [...], "answer": "..."}`` format that
``crystal-metrics`` consumes.

The model is reached through an **OpenAI-compatible** chat endpoint, so the SAME
script works for both backends — only ``--base-url`` and ``--model`` change:

    # Ollama  (default port 11434)
    ollama serve &
    ollama pull qwen2.5vl:7b
    python crystal_inference.py --model qwen2.5vl:7b \
        --base-url http://localhost:11434/v1

    # vLLM  (OpenAI-compatible server)
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 &
    python crystal_inference.py --model Qwen/Qwen2.5-VL-7B-Instruct \
        --base-url http://localhost:8000/v1

Dependencies: openai, pillow, datasets, tqdm.
"""

import argparse
import base64
import io
import json
import os
import re
import sys

# --------------------------------------------------------------------------
# Prompt (verbatim CRYSTAL reasoning prompt). {USER_INSTRUCTION} is filled in
# per sample with the question + any answer options.
# --------------------------------------------------------------------------
PROMPT = """
You are a vision-language model. First, analyze the provided image(s) and any user text silently. Do NOT reveal your internal reasoning.

Return ONLY a single, valid JSON object with this exact schema:
{"reasoning_steps": [], "answer": ""}

Rules for "reasoning_steps":
- Decide the number of steps based on task complexity; include enough to make the answer evident without filler.
- Include some inference from visual information, always anchored to visible cues.
- Write single-clause sentences, each adding a new, directly checkable fact or cue-based inference.
- You may include cautious, visually grounded commonsense using words such as “appears”, “suggests”, or “likely”, but always anchor it to visible cues (lighting/shadows; perspective/vanishing lines/horizon/tilt; scale/relative size; focus/DOF; parallax; occlusion/contact shadows; reflections/transparency; material/texture; symmetry/patterns/alignment; position/orientation/foreground–background; density/motion cues; human pose/gaze/gesture; interactions/affordances; object state; physics plausibility; signage/text/logos/typography; numbers/units; plots/charts: type, axes/ticks/units, scale (lin/log), legend↔series, gridlines/baseline, error bars/CI, trendlines, outliers/binning, colorbar; maps: scale bar, north arrow; math/geometry: labels/givens, unit checks, angle rules, Pythagorean, distance/slope, transformations, area/volume, circle theorems, trig (incl. sine/cosine laws), vectors, systems/quadratics, combinatorics, logs/exponents, probability/statistics, exact forms, conversions, plots, graphs, math equations, diagrams).
- No multi-sentence items. No internal monologue.

Rules for "answer":
- Provide the final answer grounded strictly in visible content and given text.
- If information is missing or ambiguous, set "answer" to "insufficient information" and include steps noting what is missing (e.g., “Noted the license plate is unreadable due to blur.”).
- Multiple-choice: if options have letters, return only the single best LETTER (e.g., "B"); if unlabeled, return the exact option text verbatim.
- Numeric: include required units; obey requested rounding; otherwise give exact/simplest form.

What to notice in steps (express as sentences, not labels):
- Objects & attributes (classes, colors, materials, states), logos/brands if clearly visible.
- Positions & spatial relations (left/right/above/below/front/behind, proximity, alignment, orientation, foreground/background).
- Depth cues (relative size, position in frame vs. horizon, sharpness/detail, shadow contact, occlusion order).
- Scene & lighting/time cues (indoor/outdoor, daylight vs. night, weather indications, activity/no-activity).
- Occlusion effects and how they affect certainty.
- Text/OCR with exact casing/punctuation (“Read text: ‘SPEED LIMIT 25’. ”).
- Counts & quantities for distinct instances; approximate only if visually justified.
- Graphics/plots/diagrams: axes, ticks, units, legends; read exact values rather than guessing.

Output formatting:
- Output only the JSON object. No extra keys, comments, code fences, or prose.
- Use double quotes for all strings; no trailing commas; any valid JSON whitespace is acceptable.

Now follow the user instruction:

User instruction: {USER_INSTRUCTION}
"""

DEFAULT_ANSWER = {"reasoning_steps": [], "answer": "insufficient information"}


# --------------------------------------------------------------------------
# Image encoding
# --------------------------------------------------------------------------
def pil_image_to_data_uri(img) -> str:
    """Encode a PIL image to a data URI, preserving format (PNG fallback)."""
    buffered = io.BytesIO()
    img_fmt = img.format if img.format else "PNG"
    img.save(buffered, format=img_fmt)
    b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/{img_fmt.lower()};base64,{b64}"


def pil_image_to_data_uri_resized(img, max_side: int = 768, fmt: str = "PNG") -> str:
    """Encode a downscaled copy (fallback when the server rejects large images)."""
    w, h = img.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    if fmt.upper() == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format=fmt.upper())
    b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


# --------------------------------------------------------------------------
# Robust JSON coercion + schema enforcement
# --------------------------------------------------------------------------
def extract_json_like(text: str) -> str:
    """Pull the substring between the first '{' and last '}', stripping fences."""
    text = text.replace("```", "")
    text = re.sub(r"^\s*json\s*\n", "", text, flags=re.IGNORECASE)
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first:last + 1]
    return text


def remove_invalid_control_chars(s: str) -> str:
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", s)


def strip_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)


def coerce_to_valid_json(text: str, default=None):
    """Best-effort parse of model output into JSON. Returns (obj, raw_used)."""
    if default is None:
        default = dict(DEFAULT_ANSWER)
    try:
        return json.loads(text), text
    except Exception:
        pass
    chunk = extract_json_like(text)
    chunk = remove_invalid_control_chars(chunk)
    chunk = strip_trailing_commas(chunk).strip()
    try:
        return json.loads(chunk), chunk
    except Exception:
        return default, chunk


def ensure_schema(obj):
    """Force the dict to exactly {"reasoning_steps": list[str], "answer": str}."""
    if not isinstance(obj, dict):
        return dict(DEFAULT_ANSWER), True

    corrected = False
    rs = obj.get("reasoning_steps", [])
    if not isinstance(rs, list):
        rs = [str(rs)]
        corrected = True
    else:
        new_rs = []
        for it in rs:
            if isinstance(it, str):
                new_rs.append(it)
            else:
                new_rs.append(str(it))
                corrected = True
        rs = new_rs

    ans = obj.get("answer", "insufficient information")
    if not isinstance(ans, str):
        ans = str(ans)
        corrected = True

    fixed = {"reasoning_steps": rs, "answer": ans}
    if set(obj.keys()) != set(fixed.keys()):
        corrected = True
    return fixed, corrected


def format_user_instruction(item) -> str:
    """Build the user instruction from a dataset item (question + options)."""
    if item.get("choices"):
        options = item["choices"]
    elif item.get("options"):
        options = item["options"]
    else:
        options = []
    formatted = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(options))
    return f"{item['question']}\n\n{formatted}".strip()


def build_messages(prompt_filled: str, image_uri: str):
    """OpenAI-compatible multimodal message payload."""
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_filled},
            {"type": "image_url", "image_url": {"url": image_uri}},
        ],
    }]


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def run(args) -> int:
    from openai import OpenAI
    from tqdm import tqdm

    # Load the dataset: a HF hub name (default) or a local load_from_disk path.
    if os.path.isdir(args.dataset):
        from datasets import load_from_disk
        ds = load_from_disk(args.dataset)
        data = ds[args.split] if args.split in ds else ds
    else:
        from datasets import load_dataset
        data = load_dataset(args.dataset, split=args.split)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    os.makedirs(args.output_dir, exist_ok=True)
    error_log = os.path.join(args.output_dir, "errors.log")

    total = len(data)
    end = total if args.limit is None else min(total, args.start + args.limit)
    indices = range(args.start, end)

    for idx in tqdm(indices, desc=f"{args.model}"):
        out_path = os.path.join(args.output_dir, f"{idx}.json")
        if os.path.exists(out_path):
            continue  # resumable

        item = data[idx]
        user_instruction = format_user_instruction(item)
        prompt_filled = PROMPT.replace("{USER_INSTRUCTION}", user_instruction)

        # Primary call, then a resized-image retry.
        response = None
        for uri in (pil_image_to_data_uri(item["image"]),
                    pil_image_to_data_uri_resized(item["image"], max_side=args.max_side)):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=build_messages(prompt_filled, uri),
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                break
            except Exception as e:
                last_error = e
                continue

        if response is None:
            with open(error_log, "a", encoding="utf-8") as ef:
                ef.write(f"[idx={idx}] API error: {repr(last_error)}\n")
            obj = dict(DEFAULT_ANSWER)
        else:
            raw = response.choices[0].message.content if response.choices else ""
            obj, _ = coerce_to_valid_json(raw)
            obj, _ = ensure_schema(obj)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    print(f"Done. Predictions in {args.output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Model-agnostic CRYSTAL inference (Ollama / vLLM).")
    p.add_argument("--model", required=True, help="Model name served by the endpoint.")
    p.add_argument("--base-url", default="http://localhost:11434/v1",
                   help="OpenAI-compatible endpoint (Ollama :11434/v1, vLLM :8000/v1).")
    p.add_argument("--api-key", default="EMPTY", help="API key (unused by Ollama/vLLM, but required).")
    p.add_argument("--dataset", default="waybarrios/CRYSTAL",
                   help="HF hub dataset name or local load_from_disk path.")
    p.add_argument("--split", default="test", help="Dataset split.")
    p.add_argument("--output-dir", default="predictions", help="Where to write per-sample JSON.")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-side", type=int, default=1024, help="Resize cap for the retry image.")
    p.add_argument("--start", type=int, default=0, help="Start index.")
    p.add_argument("--limit", type=int, default=None, help="Max samples to run (default: all).")
    return p


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
