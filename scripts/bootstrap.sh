#!/usr/bin/env bash
# bootstrap.sh — one-command dev-env setup for agi-hpc (Linux/macOS)
# Usage: source scripts/bootstrap.sh

GREEN='\033[0;32m'  RED='\033[0;31m'  YELLOW='\033[1;33m'  NC='\033[0m'
banner()  { printf "\n${GREEN}=== SUCCESS: %s ===${NC}\n\n" "$1"; }
fail()    { printf "\n${RED}=== FAILED: %s ===${NC}\n\n" "$1" >&2; _bootstrap_ok=false; }
info()    { printf "${YELLOW}>>> %s${NC}\n" "$1"; }

_bootstrap_ok=true

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

# 1. Check Python >= 3.10
info "Checking Python version..."
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    fail "Python not found — install Python 3.10+"
fi

if $_bootstrap_ok; then
    PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$("$PYTHON" -c 'import sys; print(sys.version_info.major)')
    PY_MINOR=$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')
    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
        fail "Python >= 3.10 required (found $PY_VERSION)"
    else
        banner "Python $PY_VERSION"
    fi
fi

# 2. Create or reuse venv
if $_bootstrap_ok; then
    VENV_DIR="$REPO_ROOT/.venv"
    if [ -d "$VENV_DIR" ]; then
        info "Reusing existing venv at .venv/"
    else
        info "Creating venv..."
        if ! "$PYTHON" -m venv "$VENV_DIR"; then
            fail "Could not create venv"
        else
            banner "venv created"
        fi
    fi
fi

if $_bootstrap_ok; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    banner "venv activated ($(python3 --version))"
fi

# 3. Install dependencies
if $_bootstrap_ok; then
    info "Installing project dependencies (core + dev)..."
    if pip install --upgrade pip --quiet && pip install -e "${REPO_ROOT}[dev]" --quiet; then
        banner "Dependencies installed"
    else
        fail "pip install failed"
    fi
fi

# 4. Smoke test
if $_bootstrap_ok; then
    info "Running smoke tests..."
    if pytest "$REPO_ROOT/tests/test_basic.py" -v; then
        banner "All smoke tests passed — you're ready to go!"
    else
        fail "Smoke tests failed — check output above"
    fi
fi

if ! $_bootstrap_ok; then
    printf "\n${RED}Bootstrap did not complete successfully. Check errors above.${NC}\n"
fi
unset _bootstrap_ok
