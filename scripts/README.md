# scripts/bootstrap.sh

One-command dev-env setup for Linux and macOS.

## Usage

```bash
source scripts/bootstrap.sh
```

The script must be **sourced** (not executed) so the virtualenv stays active in your shell.

## What it does

1. Verifies Python >= 3.10 is installed
2. Creates a `.venv` virtualenv (reuses it if one already exists)
3. Activates the virtualenv
4. Installs all project dependencies from `pyproject.toml` (core + dev extras)
5. Runs `pytest tests/test_basic.py` as a smoke check

Each step prints a colored **SUCCESS** or **FAILED** banner. On failure the script stops progressing but keeps the terminal open so you can read the error.

## When to rerun

- After a fresh `git clone`
- After `pyproject.toml` dependencies change
- After deleting the `.venv` directory
