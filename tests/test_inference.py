"""Unit tests for the inference helpers (no network, no model, no GPU)."""

import importlib.util
import os

import pytest

# Load inference/crystal_inference.py by path (it lives outside the package).
_INFER_PATH = os.path.join(os.path.dirname(__file__), "..", "inference", "crystal_inference.py")
_spec = importlib.util.spec_from_file_location("crystal_inference", _INFER_PATH)
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)


# --- prompt -----------------------------------------------------------------
def test_prompt_has_schema_and_placeholder():
    assert '{"reasoning_steps": [], "answer": ""}' in ci.PROMPT
    assert "{USER_INSTRUCTION}" in ci.PROMPT


def test_format_user_instruction_with_choices():
    item = {"question": "Which is smallest?", "choices": ["left", "middle", "right"]}
    out = ci.format_user_instruction(item)
    assert out.startswith("Which is smallest?")
    assert "A) left" in out and "B) middle" in out and "C) right" in out


def test_format_user_instruction_open_ended():
    item = {"question": "How many cats?"}
    assert ci.format_user_instruction(item) == "How many cats?"


# --- JSON coercion ----------------------------------------------------------
def test_coerce_clean_json():
    obj, _ = ci.coerce_to_valid_json('{"reasoning_steps": ["a"], "answer": "B"}')
    assert obj == {"reasoning_steps": ["a"], "answer": "B"}


def test_coerce_with_code_fence_and_prose():
    raw = 'Here you go:\n```json\n{"reasoning_steps": ["a", "b"], "answer": "C"}\n```'
    obj, _ = ci.coerce_to_valid_json(raw)
    assert obj["answer"] == "C"
    assert obj["reasoning_steps"] == ["a", "b"]


def test_coerce_trailing_comma():
    obj, _ = ci.coerce_to_valid_json('{"reasoning_steps": ["a",], "answer": "B",}')
    assert obj["answer"] == "B"


def test_coerce_garbage_returns_default():
    obj, _ = ci.coerce_to_valid_json("the model said nothing useful")
    assert obj == ci.DEFAULT_ANSWER


# --- schema enforcement -----------------------------------------------------
def test_ensure_schema_coerces_nonstring_steps():
    fixed, corrected = ci.ensure_schema({"reasoning_steps": [1, 2], "answer": 5})
    assert fixed == {"reasoning_steps": ["1", "2"], "answer": "5"}
    assert corrected is True


def test_ensure_schema_drops_extra_keys():
    fixed, corrected = ci.ensure_schema(
        {"reasoning_steps": ["a"], "answer": "B", "extra": 1}
    )
    assert set(fixed.keys()) == {"reasoning_steps", "answer"}
    assert corrected is True


def test_ensure_schema_valid_unchanged():
    fixed, corrected = ci.ensure_schema({"reasoning_steps": ["a"], "answer": "B"})
    assert fixed == {"reasoning_steps": ["a"], "answer": "B"}
    assert corrected is False


# --- message payload --------------------------------------------------------
def test_build_messages_shape():
    msgs = ci.build_messages("hello", "data:image/png;base64,xxx")
    assert msgs[0]["role"] == "user"
    types = [part["type"] for part in msgs[0]["content"]]
    assert types == ["text", "image_url"]
