"""Tests for agi.primer.validator.

Covers extract_code parsing and validate() against tiny synthetic tasks
— both a correct implementation (all_pass=True) and broken ones
(compile error, wrong output, no transform()).
"""

from __future__ import annotations

from agi.primer.validator import extract_code, validate

# ── extract_code ─────────────────────────────────────────────────


def test_extract_code_from_json():
    src = '{"code": "def transform(g):\\n    return g", "rule": "identity"}'
    assert "def transform" in extract_code(src)


def test_extract_code_from_markdown_fence():
    src = (
        "Here is a solution:\n\n```python\ndef transform(g):\n    return g\n```\nDone."
    )
    code = extract_code(src)
    assert code.startswith("def transform")
    assert code.endswith("return g") or code.endswith("return g\n")


def test_extract_code_from_raw_python():
    src = "some prose\ndef transform(g):\n    return g"
    code = extract_code(src)
    assert code.startswith("def transform")


def test_extract_code_missing():
    assert extract_code("no code here") == ""


# ── validate ─────────────────────────────────────────────────────


_IDENTITY_TASK = {
    "train": [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
        {"input": [[0]], "output": [[0]]},
    ],
    "test": [],
}


def test_validate_correct_identity():
    code = "def transform(g):\n    return g"
    r = validate(code, _IDENTITY_TASK)
    assert r.all_pass
    assert len(r.per_example) == 2
    assert all(e["correct"] for e in r.per_example)


def test_validate_wrong_output():
    code = "def transform(g):\n    return [[0]]"  # always returns [[0]]
    r = validate(code, _IDENTITY_TASK)
    assert not r.all_pass
    assert any(not e["correct"] for e in r.per_example)
    assert r.diagnostic


def test_validate_compile_error():
    code = "def transform(g):\n    return g undefined_syntax"
    r = validate(code, _IDENTITY_TASK)
    assert not r.all_pass
    assert "compile error" in r.diagnostic.lower() or "syntax" in r.diagnostic.lower()


def test_validate_no_transform():
    code = "def other_name(g):\n    return g"
    r = validate(code, _IDENTITY_TASK)
    assert not r.all_pass


def test_validate_runtime_error():
    code = "def transform(g):\n    raise ValueError('boom')"
    r = validate(code, _IDENTITY_TASK)
    assert not r.all_pass
    assert r.per_example
    assert not r.per_example[0]["correct"]
    assert "ValueError" in r.per_example[0].get("reason", "")


def test_validate_numpy_returned_ok():
    # Using numpy internally is fine; result should be coerced to list-of-lists
    code = "import numpy as np\n" "def transform(g):\n" "    return np.array(g)\n"
    r = validate(code, _IDENTITY_TASK)
    assert r.all_pass
