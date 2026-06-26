# Installation

`crystal-metrics` requires **Python 3.8+**. Published on PyPI:
https://pypi.org/project/crystal-metrics/

## From PyPI

```bash
pip install crystal-metrics          # core metrics (Match F1, Ordered F1, Accuracy)
pip install crystal-metrics[judge]   # + optional LLM judge for free-form answers
```

## From source (this repo)

```bash
git clone https://github.com/waybarrios/crystal.git
cd crystal
pip install -e .            # editable install
pip install -e ".[dev]"     # + pytest for running the test suite
```

## Dependencies

| Group | Packages | When you need it |
|-------|----------|------------------|
| core  | `numpy`, `pandas`, `torch`, `sentence-transformers`, `tqdm` | Always |
| `[judge]` | `openai` | Only to grade free-form text answers with an LLM |
| `[dev]` | `pytest` | Running tests |

The core install is enough for Match F1, Ordered Match F1, Precision, Recall, and
rule-based Accuracy. The LLM judge is **opt-in** — importing the package never
requires `openai`.

## First run downloads an embedding model

The default model is `all-distilroberta-v1` (ablation-validated, threshold
τ = 0.35). On first use it is downloaded and cached under
`~/.cache/sentence_transformers`. For offline machines, pre-download it once on a
connected machine and copy the cache.

## Verify the install

```bash
python -c "import crystal_metrics; print(crystal_metrics.__version__)"
pytest          # runs unit + parity tests
```
