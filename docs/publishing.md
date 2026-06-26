# Publishing `crystal-metrics` to PyPI

Maintainer guide. Publishing is **irreversible** — a given version can be yanked
but never re-uploaded. Bump the version in `pyproject.toml` for every release.

## 1. Prerequisites

```bash
pip install build twine
```

A PyPI account with an **API token** (https://pypi.org/manage/account/token/).
Scope it to the `crystal-metrics` project after the first upload.

## 2. Build

```bash
rm -rf dist build src/crystal_metrics.egg-info
python -m build            # writes dist/*.whl and dist/*.tar.gz
python -m twine check dist/*
```

## 3. (Recommended) Dry run on TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ crystal-metrics
```

## 4. Upload to PyPI

Keep the token out of your shell history — pass it via environment variables:

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-XXXXXXXX \
    python -m twine upload dist/*
```

## 5. Verify

```bash
pip install crystal-metrics
python -c "import crystal_metrics; print(crystal_metrics.__version__)"
```

## Releasing a new version

1. Update `version` in `pyproject.toml` and `__version__` in
   `src/crystal_metrics/__init__.py` (keep them in sync).
2. Rebuild and re-upload (steps 2 and 4).
