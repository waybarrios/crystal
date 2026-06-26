# Running inference on CRYSTAL

`inference/crystal_inference.py` runs **any** vision-language model over the
CRYSTAL benchmark and writes predictions in the format the metrics consume. It is
**model-agnostic**: the model is reached through an OpenAI-compatible chat
endpoint, so the same script works for both **Ollama** and **vLLM** — only
`--base-url` and `--model` change.

```bash
pip install -r inference/requirements.txt   # openai, pillow, datasets, tqdm
```

## Option A — Ollama

```bash
ollama serve &                      # starts the API on :11434
ollama pull qwen2.5vl:7b            # any vision model Ollama serves

python inference/crystal_inference.py \
    --model qwen2.5vl:7b \
    --base-url http://localhost:11434/v1 \
    --output-dir predictions/qwen2.5vl-7b
```

## Option B — vLLM

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 \
    --gpu-memory-utilization 0.9 &   # OpenAI-compatible server on :8000

python inference/crystal_inference.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --base-url http://localhost:8000/v1 \
    --output-dir predictions/qwen2.5vl-7b
```

That is the only difference between backends: the server you start, the
`--base-url`, and the `--model` name.

## Key options

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | (required) | Model name served by the endpoint |
| `--base-url` | `http://localhost:11434/v1` | OpenAI-compatible endpoint (Ollama / vLLM) |
| `--api-key` | `EMPTY` | API key (unused by Ollama/vLLM, but the client requires a value) |
| `--dataset` | `waybarrios/CRYSTAL` | HF hub name or a local `load_from_disk` path |
| `--split` | `test` | Dataset split |
| `--output-dir` | `predictions` | Where per-sample JSON is written |
| `--max-tokens` | `4096` | Generation cap |
| `--temperature` | `0.1` | Sampling temperature (low = near-deterministic) |
| `--max-side` | `1024` | Resize cap (px) for the retry image when the server rejects a large one |
| `--start` / `--limit` | `0` / all | Run a slice (e.g. smoke-test with `--limit 5`) |

The CRYSTAL dataset is gated — `huggingface-cli login` and request access first,
or pass a local path to `--dataset`.

## The prompt

The script sends the exact CRYSTAL reasoning prompt (defined as `PROMPT` in the
script). It instructs the model to return **only** a JSON object:

```json
{"reasoning_steps": ["single-clause grounded step", "..."], "answer": "B"}
```

`{USER_INSTRUCTION}` is replaced per sample with the question plus any
multiple-choice options (`A) ...`, `B) ...`).

## Output

One file per sample, `<output-dir>/<index>.json`:

```json
{"reasoning_steps": ["...", "..."], "answer": "C"}
```

The run is **resumable** — existing files are skipped, so you can re-run after an
interruption. Robust JSON coercion + schema enforcement guarantee every file is
valid `{reasoning_steps, answer}` even when the model adds prose or code fences.

## Then evaluate

The output directory drops straight into the metrics. Build a references file
from the dataset (`{id: {reference_steps, answer}}`) and run:

```bash
crystal-metrics evaluate predictions/qwen2.5vl-7b references.json --alpha 0.3
```

See [metrics](metrics.md) and [CLI](cli.md).
